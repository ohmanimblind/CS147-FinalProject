#!/usr/bin/env bash

BIN=./main

start=$SECONDS

for FOLD in fold{1..2}; do
	OUTDIR=/scratch/apadi089/audio/${FOLD}proc
	mkdir -p "$OUTDIR"

#	echo "Proccessing $FOLD"
	
	for WAV in /scratch/apadi089/audio/"$FOLD"/*.wav; do
		BASENAME=$(basename "$WAV" .wav)


		"$BIN" "$WAV"



		mv spectrogram.bin "$OUTDIR/${BASENAME}.bin"
#		echo " moving $BASENAME.wav -> $OUTDIR/${BASENAME}.bin "
	done
done

duration=$((SECONDS - start))
echo "Duration in seconds $duration"

echo "done"
