use loupe_server::LoupeService;
use loupe_server::proto::loupe_server::Loupe;
use loupe_server::proto::{CreateCollectionRequest, Field, SearchRequest, UpsertRequest, field};
use tonic::Request;

fn str_field(key: &str, value: &str) -> Field {
    Field {
        key: key.to_string(),
        value: Some(field::Value::StrValue(value.to_string())),
    }
}

#[tokio::test]
async fn create_upsert_search_and_explain() {
    let service = LoupeService::new();

    service
        .create_collection(Request::new(CreateCollectionRequest {
            name: "docs".to_string(),
            dim: 2,
            metric: "l2".to_string(),
        }))
        .await
        .unwrap();

    for i in 0..20u32 {
        let coordinate = i as f32;
        let category = if i % 2 == 0 { "a" } else { "b" };
        service
            .upsert(Request::new(UpsertRequest {
                collection: "docs".to_string(),
                vector: vec![coordinate, coordinate],
                fields: vec![str_field("cat", category)],
            }))
            .await
            .unwrap();
    }

    let search = service
        .search(Request::new(SearchRequest {
            collection: "docs".to_string(),
            query: vec![4.0, 4.0],
            k: 5,
            filter: "cat=a".to_string(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(!search.neighbors.is_empty());
    for neighbor in &search.neighbors {
        assert_eq!(neighbor.id % 2, 0);
    }

    let explain = service
        .explain_search(Request::new(SearchRequest {
            collection: "docs".to_string(),
            query: vec![4.0, 4.0],
            k: 5,
            filter: "cat=a".to_string(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(explain.chosen_plan.is_some());
    assert_eq!(explain.counterfactual_plans.len(), 3);
}

#[tokio::test]
async fn search_on_unknown_collection_is_an_error() {
    let service = LoupeService::new();
    let result = service
        .search(Request::new(SearchRequest {
            collection: "missing".to_string(),
            query: vec![1.0, 1.0],
            k: 5,
            filter: "cat=a".to_string(),
        }))
        .await;
    assert!(result.is_err());
}
