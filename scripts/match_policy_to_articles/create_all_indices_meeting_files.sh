sbatch create_one_index_endeavour.sh \
   --index_name seattle_agenda \
   --text_col title \
   --id_col id   \
   --file_pattern_to_index "../data/seattle_videos/*.schedule.csv" \
   --embedding_model bm25

sbatch create_one_index_endeavour.sh \
   --index_name newark_agenda \
   --text_col title \
   --id_col id   \
   --file_pattern_to_index "../data/newark_videos/*.schedule.csv" \
   --embedding_model bm25

sbatch create_one_index_endeavour.sh \
   --index_name jacksonville_agenda \
   --text_col title \
   --id_col id   \
   --file_pattern_to_index "../data/jacksonville_videos/*.schedule.csv" \
   --embedding_model bm25

sbatch create_one_index_endeavour.sh \
   --index_name fortworth_agenda \
   --text_col title \
   --id_col id   \
   --file_pattern_to_index "../data/fortworth_videos/*.schedule.csv" \
   --embedding_model bm25

sbatch create_one_index_endeavour.sh \
   --index_name durham_agenda \
   --text_col title \
   --id_col id   \
   --file_pattern_to_index "../data/durham_videos/*.schedule.csv" \
   --embedding_model bm25

sbatch create_one_index_endeavour.sh \
   --index_name denver_agenda \
   --text_col title \
   --id_col id   \
   --file_pattern_to_index "../data/denver_videos/*.schedule.csv" \
   --embedding_model bm25

