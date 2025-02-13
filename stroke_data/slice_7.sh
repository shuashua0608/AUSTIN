
for d in Audio_44k_trim/* ; do
    x=${d%.wav}
    fn="${x##*/}"
    mkdir fix_len_audio_trim/segments/${fn}
    # DUR=$(ffprobe -i ${d} -show_entries format=duration -v quiet -of csv="p=0" | awk '{print $1 }')
    # ST1=$(echo "$DUR"/ 7 | bc) 
    ffmpeg -i ${d} -f segment -segment_time 7 -segment_start_number 1 fix_len_audio_trim/segments/${fn}/%02d.wav
    rm fix_len_audio_trim/segments/${fn}/08.wav
    rm fix_len_audio_trim/segments/${fn}/09.wav
    rm fix_len_audio_trim/segments/${fn}/10.wav
    rm fix_len_audio_trim/segments/${fn}/11.wav
    rm fix_len_audio_trim/segments/${fn}/12.wav
done
