# Seq2Seq Model with Dot‑Product Attention (Educational Implementation)

Author: Keith Paschal  
Python • PyTorch • NLP • Seq2Seq • Attention

---------------------------------------------------------------------

## Overview

This repository contains a transparent, step‑by‑step implementation of a
sequence‑to‑sequence (seq2seq) neural network with dot‑product attention,
built using PyTorch’s low‑level APIs.

The model demonstrates how encoder and decoder RNN cells work internally,
how attention scores are computed, and how context vectors are applied
during decoding. All logic is contained in a single file for clarity.

---------------------------------------------------------------------

## Features

- Custom Encoder built with nn.RNNCell
- Manual dot‑product attention implementation
- Autoregressive Decoder with attention applied at each timestep
- Clear intermediate printouts for learning (hidden states, scores, weights)
- Fully self‑contained in: seq2seq.py

---------------------------------------------------------------------

## Global DEBUG Flag

This project uses a module‑level `DEBUG` flag that controls whether
internal calculations (hidden states, attention weights, matrix shapes)
are printed during execution.

- By default: `DEBUG = False`
- When running the script directly, debug mode is enabled by:
```Python
if __name__ == "__main__":
    DEBUG = True
    main()
```
This ensures:
- Importing the module does **not** activate debug mode  
- Running `python seq2seq.py` **does** activate debug mode  
- The rest of the code can check the global DEBUG variable safely 

---------------------------------------------------------------------

## File Structure

seq2seq.py  
README.md

---------------------------------------------------------------------

## Running the Code

Run the file directly:

python seq2seq.py

This executes a forward pass and displays internal tensor operations
(step‑by‑step shapes, attention weights, and context vector calculations).

---------------------------------------------------------------------

## Requirements

Python 3.9+  
PyTorch  
NumPy

Install dependencies:

pip install torch numpy

---------------------------------------------------------------------

## Purpose

This project is ideal for:

- Understanding how seq2seq models operate internally
- Learning dot‑product attention computation
- Studying matrix operations in neural networks
- Using RNNCell‑based encoder/decoder logic instead of high‑level layers

---------------------------------------------------------------------

## License

This project is provided for educational and research purposes. You may modify,
extend, or adapt it. See `LICENSE` for details (MIT recommended).