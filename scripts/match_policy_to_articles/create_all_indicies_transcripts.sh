sbatch create_one_index_endeavour.sh \
   --index_name seattle_transcript \
   --text_col text \
   --id_col id   \
   --file_pattern_to_index "../data/seattle_transcribed_files/*.transcribed.json" \
   --embedding_model bm25

sbatch create_one_index_endeavour.sh \
   --index_name newark_transcript \
   --text_col text \
   --id_col id   \
   --file_pattern_to_index "../data/newark_transcribed_files/*.transcribed.json" \
   --embedding_model bm25
    
sbatch create_one_index_endeavour.sh \
   --index_name jacksonville_transcript \
   --text_col text \
   --id_col id   \
   --file_pattern_to_index "../data/jacksonville_transcribed_files/*.transcribed.json" \
   --embedding_model bm25

sbatch create_one_index_endeavour.sh \
   --index_name fortworth_transcript \
   --text_col text \
   --id_col id   \
   --file_pattern_to_index "../data/fortworth_transcribed_files/*.transcribed.json" \
   --embedding_model bm25

sbatch create_one_index_endeavour.sh \
   --index_name durham_transcript \
   --text_col text \
   --id_col id   \
   --file_pattern_to_index "../data/durham_transcribed_files/*.transcribed.json" \
   --embedding_model bm25

sbatch create_one_index_endeavour.sh \
   --index_name denver_transcript \
   --text_col text \
   --id_col id   \
   --file_pattern_to_index "../data/denver_transcribed_files/*.transcribed.json" \
   --embedding_model bm25

