
if [ ! -f "food-11.zip" ]; then
    curl --user datasets@mmspgdata.epfl.ch:ohsh9jah4T ftp://tremplin.epfl.ch/FoodImage/Food-11.zip --output food-11.zip
fi
if [ ! -f "food-11.zip" ]; then
    echo "Error: could not download the dataset"
    exit 1
fi
if [ ! -d "data" ]; then
    unzip -qo food-11.zip -d data
fi
if [ ! -d "data" ]; then
    echo "Error: could not extract the dataset"
    exit 1
fi
echo "Dataset downloaded and extracted"



