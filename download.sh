
echo "Downloading SUN dataset..."
# wget -c http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
echo "Extracting..."
tar xzf SUN2012.tar.gz
echo "Resizing images..."
find . -name "*.jpg" | xargs mogrify -resize 128x128!
echo "Saturating..."
find . -name "*.jpg" | xargs -I % convert % -colorspace HSL -channel Saturation -negate -evaluate multiply 0.8 -negate -colorspace RGB %
echo "Remove corrupted images..."
rm SUN2012/Images/d/dirt_track/sun_banvbkentoawqquo.jpg
rm SUN2012/Images/c/car_dealership/sun_bfzdurixkybkmlmp.jpg
python3 remove_invalid.py SUN2012/
echo "Done!"
