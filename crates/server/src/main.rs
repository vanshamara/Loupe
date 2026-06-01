use loupe_server::LoupeService;
use loupe_server::proto::loupe_server::LoupeServer;

/// Serves the Loupe gRPC API on a local port.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let address = "127.0.0.1:50051".parse()?;
    println!("loupe-server listening on {address}");
    tonic::transport::Server::builder()
        .add_service(LoupeServer::new(LoupeService::new()))
        .serve(address)
        .await?;
    Ok(())
}
