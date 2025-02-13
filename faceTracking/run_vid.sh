for d in ../vid_3DDFA_raw_all/* ; do
    x=${d%.wav}
    fn="${x##*/}"
    echo ${fn:0:4}
    python framerun.py -v ${fn:0:4}
done