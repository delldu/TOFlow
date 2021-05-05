python evaluate.py \
	--task zoom \
	--dataDir tiny/vimeo_septuplet/sequences_blur \
	--ex_dataDir tiny/vimeo_septuplet/sequences_blur \
	--pathlist tiny/vimeo_septuplet/sep_testlist.txt \
	--model toflow_models/sr.pkl \
	--gpuID 0

python evaluate.py \
	--task slow \
	--dataDir tiny/vimeo_triplet/sequences \
	--pathlist tiny/vimeo_triplet/tri_testlist.txt \
	--model toflow_models/interp.pkl --gpuID 0

python evaluate.py \
	--task clean \
	--dataDir tiny/vimeo_septuplet/sequences_with_noise \
	--ex_dataDir tiny/vimeo_septuplet/sequences_with_noise \
	--pathlist tiny/vimeo_septuplet/sep_testlist.txt \
	--model toflow_models/denoise.pkl \
	--gpuID 0
