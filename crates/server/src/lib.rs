//! gRPC server: exposes the engine over the network so a client can create a
//! collection, upsert vectors, and run search and EXPLAIN remotely.

// Every gRPC handler must return `Result<_, tonic::Status>`, and `Status` is large.
#![allow(clippy::result_large_err)]

#[allow(clippy::all, clippy::pedantic, clippy::nursery)]
pub mod proto {
    tonic::include_proto!("loupe");
}

use std::collections::HashMap;
use std::sync::Mutex;

use loupe_distance::Metric;
use loupe_engine::{Collection, Value, parse_filter};
use tonic::{Request, Response, Status};

use proto::loupe_server::Loupe;
use proto::{
    CreateCollectionRequest, CreateCollectionResponse, ExplainSearchResponse, SearchRequest,
    SearchResponse, UpsertRequest, UpsertResponse,
};

/// Holds the collections served over gRPC. Each request locks the map, does its
/// (synchronous) work, and releases it.
#[derive(Default)]
pub struct LoupeService {
    collections: Mutex<HashMap<String, Collection>>,
}

impl LoupeService {
    pub fn new() -> Self {
        Self::default()
    }
}

#[tonic::async_trait]
impl Loupe for LoupeService {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let request = request.into_inner();
        let metric = parse_metric(&request.metric)?;
        self.collections
            .lock()
            .unwrap()
            .insert(request.name, Collection::new(request.dim as usize, metric));
        Ok(Response::new(CreateCollectionResponse {}))
    }

    async fn upsert(
        &self,
        request: Request<UpsertRequest>,
    ) -> Result<Response<UpsertResponse>, Status> {
        let request = request.into_inner();
        let fields = to_fields(request.fields)?;
        let field_refs: Vec<(&str, Value)> = fields
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        let id = {
            let mut collections = self.collections.lock().unwrap();
            let collection = collections
                .get_mut(&request.collection)
                .ok_or_else(|| Status::not_found("unknown collection"))?;
            collection.insert(&request.vector, &field_refs)
        };
        Ok(Response::new(UpsertResponse { id }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let request = request.into_inner();
        let expr = parse_filter(&request.filter).map_err(Status::invalid_argument)?;
        let neighbors = {
            let collections = self.collections.lock().unwrap();
            let collection = collections
                .get(&request.collection)
                .ok_or_else(|| Status::not_found("unknown collection"))?;
            let (results, _) = collection.search(&request.query, &expr, request.k as usize);
            results
                .iter()
                .map(|n| proto::Neighbor {
                    id: n.id,
                    distance: n.distance,
                })
                .collect()
        };
        Ok(Response::new(SearchResponse { neighbors }))
    }

    async fn explain_search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<ExplainSearchResponse>, Status> {
        let request = request.into_inner();
        let expr = parse_filter(&request.filter).map_err(Status::invalid_argument)?;
        let explain = {
            let collections = self.collections.lock().unwrap();
            let collection = collections
                .get(&request.collection)
                .ok_or_else(|| Status::not_found("unknown collection"))?;
            let (_, explain) = collection.explain_search(&request.query, &expr, request.k as usize);
            explain
        };
        Ok(Response::new(to_proto_explain(&explain)))
    }
}

fn parse_metric(name: &str) -> Result<Metric, Status> {
    match name.to_lowercase().as_str() {
        "l2" => Ok(Metric::L2),
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        other => Err(Status::invalid_argument(format!("unknown metric: {other}"))),
    }
}

fn to_fields(fields: Vec<proto::Field>) -> Result<Vec<(String, Value)>, Status> {
    fields
        .into_iter()
        .map(|field| {
            let value = field
                .value
                .ok_or_else(|| Status::invalid_argument("field has no value"))?;
            let value = match value {
                proto::field::Value::IntValue(int) => Value::Int(int),
                proto::field::Value::StrValue(text) => Value::Str(text),
            };
            Ok((field.key, value))
        })
        .collect()
}

fn to_proto_explain(explain: &loupe_engine::ExplainSearch) -> ExplainSearchResponse {
    ExplainSearchResponse {
        k: explain.query_summary.k as u32,
        matching_count: explain.query_summary.matching_count,
        selectivity: explain.query_summary.selectivity,
        correlation: explain.query_summary.correlation.clone(),
        chosen_plan: Some(to_proto_plan(&explain.chosen_plan)),
        counterfactual_plans: explain
            .counterfactual_plans
            .iter()
            .map(to_proto_plan)
            .collect(),
        risk_flags: explain.risk_flags.clone(),
        recommendations: explain.recommendations.clone(),
    }
}

fn to_proto_plan(plan: &loupe_engine::PlanReport) -> proto::PlanReport {
    proto::PlanReport {
        strategy: plan.strategy.clone(),
        chosen: plan.chosen,
        distance_computations: plan.distance_computations,
        returned: plan.returned as u64,
        recall_proxy: plan.recall_proxy,
    }
}
