import os
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset import NodePropPredDataset

def download_datasets():
    # Create data directories if they don't exist
    os.makedirs("data/raw/tgb", exist_ok=True)
    os.makedirs("data/processed/tgb", exist_ok=True)

    # Download link prediction datasets
    print("Downloading link prediction datasets...")
    link_datasets = ["tgbl-wiki", "tgbl-review", "tgbl-coin"]
    for name in link_datasets:
        try:
            print(f"\nDownloading {name}...")
            dataset = LinkPropPredDataset(name=name, root="data/raw/tgb")
            print(f"Successfully downloaded {name}")
            
            # Get some basic statistics
            data = dataset.full_data
            print(f"Number of nodes: {data.num_nodes}")
            print(f"Number of edges: {data.num_edges}")
            print(f"Time span: {data.t_span} seconds")
        except Exception as e:
            print(f"Error downloading {name}: {str(e)}")
    
    # Download node prediction datasets
    print("\nDownloading node prediction datasets...")
    node_datasets = ["tgbn-trade", "tgbn-flight"]
    for name in node_datasets:
        try:
            print(f"\nDownloading {name}...")
            dataset = NodePropPredDataset(name=name, root="data/raw/tgb")
            print(f"Successfully downloaded {name}")
            
            # Get some basic statistics
            data = dataset.full_data
            print(f"Number of nodes: {data.num_nodes}")
            print(f"Number of edges: {data.num_edges}")
            print(f"Time span: {data.t_span} seconds")
        except Exception as e:
            print(f"Error downloading {name}: {str(e)}")

if __name__ == "__main__":
    download_datasets() 