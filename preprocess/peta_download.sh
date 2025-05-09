#!/bin/bash

echo "Downloading PETA.zip from GitHub Release..."
wget https://github.com/jungseoik/Distance_Threat_Pedestrian/releases/download/v1.0/PETA.zip -O PETA.zip

echo "Unzipping PETA.zip into ./assets directory..."
unzip -q PETA.zip -d assets/

echo "Done. Files extracted to ./assets"
