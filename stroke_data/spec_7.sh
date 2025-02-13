
for d in Raw_Video/* ; do
    x=${d%.MOV}
    fn="${x##*/}"
    echo $fn
    python spec_7.py --path ${fn}
done
