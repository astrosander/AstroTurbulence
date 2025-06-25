#!/bin/bash

# Path to join_vtk++ executable
join_vtk_path="$HOME/Downloads/athena/vis/vtk/join_vtk++"

# Determine the max frame number from .vtk files
max_frame=$(find . -maxdepth 1 -name "ma10_vtk.block*.out2.[0-9][0-9][0-9][0-9][0-9].vtk" |
            sed -E 's/.*\.out2\.([0-9]{5})\.vtk/\1/' | sort -n | tail -n 1)

echo "Max frame: $max_frame"

# Loop over all frame numbers up to max_frame
for frame in $(seq -w 0 "$max_frame"); do
    # Construct output filename
    output_file="ma10_vtk.${frame}.vtk"

    # Find all VTK files for this frame
    vtk_files=$(find . -maxdepth 1 -name "ma10_vtk.block*.out2.${frame}.vtk" | sort)

    # If there are any files to join
    if [ -n "$vtk_files" ]; then
        echo "Joining frame $frame..."
        $join_vtk_path -o "$output_file" $vtk_files
        echo "Output: $output_file"
    else
        echo "No VTK files found for frame $frame"
    fi
done
