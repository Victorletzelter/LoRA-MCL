#!/bin/bash
cd ./wandb
# Check if at least one folder is provided
if [ "$#" -eq 0 ]; then
	    echo "Usage: $0 /path/to/wandb/run1 /path/to/wandb/run2 ..."
	        exit 1
fi

# Loop over each folder and sync
for folder in "$@"; do
	    if [ -d "$folder" ]; then
		            echo "Syncing $folder..."
			            wandb sync "$folder" -p llavaMCL
				        else
						        echo "Warning: '$folder' is not a directory or does not exist."
							    fi
						    done

