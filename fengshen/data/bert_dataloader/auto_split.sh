files=`find $1 -type f -size +1024M`

for p in $files
do
echo "processing $p"
name=`basename $p .json`
file=`dirname $p`
split -a 2 -C 300M $p $file/$name- && ls|grep -E "(-[a-zA-Z]{2})" |xargs -n1 -i{} mv {} {}.json
rm -f $p
done