for win_size in 11 21 41 71 101; do
    for num_gestures in 5 10 15 20 40; do
        python src/kmeans/kmeans_rtMRI.py --win_size $win_size --num_gestures $num_gestures
    done
done
