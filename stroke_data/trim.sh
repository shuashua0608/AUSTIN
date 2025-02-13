for d in Raw_Video/* ; do
    x=${d%.MOV}
    fn="${x##*/}"
    echo $fn
    python trim_white.py ${fn}
done
