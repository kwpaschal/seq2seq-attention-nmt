"""
Seq2Seq Neural Machine Translation Model with Dot-Product Attention
Author: Keith Paschal

This project implements:
- A custom Encoder built with nn.RNNCell
- A Decoder using step-by-step token generation
- A Dot-Product Attention mechanism
- Full forward pass illustrating all matrix operations

The code is designed for instructional and experimentation purposes in NLP.
"""
import torch
import torch.nn as nn

class DotAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters needed for dot attention

    def forward(self, decoder_state, encoder_outputs):
        """
        :param decoder_state:   [batch_size, hidden_dim]           (s_t)
        :param encoder_outputs: [batch_size, src_len, hidden_dim]  (h_1..h_N)
        :return:
            context:      [batch_size, hidden_dim]
            attn_weights: [batch_size, src_len]
        """
        B, S, H = encoder_outputs.shape  #B = batch, S = source length, H = hidden dims

        # 1) Dot product for attention scores => [batch_size, src_len]
        #    (s_t^T * h_i for each i)
        #scores =torch.bmm(encoder_outputs, decoder_state.unsqueeze(2)).squeeze(2)
        scores = torch.zeros(B,S)
        for b in range(B):
            s_t = decoder_state[b] #This is [H]
            if DEBUG:
                print(f'Batch {b} decoder_state s_t (shape {tuple(s_t.shape)})')
                print(f'Batch {b} computing dot products s_t^T h_i for i=0..{S-1}')
            for i in range(S):
                h_i = encoder_outputs[b,i]
                dot_bi = torch.dot(s_t, h_i) #torch deals with the transposing thingy
                scores[b,i] = dot_bi
                if DEBUG:
                    s_list  = [round(float(x), 4) for x in s_t.tolist()]
                    h_list  = [round(float(x), 4) for x in h_i.tolist()]
                    dot_val = round(float(dot_bi.item()), 6)
                    print(f"  i={i}:  s_t · h_{i} = {s_list[:4]} · {h_list[:4]} = {dot_val}")


        # 2) Softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=1)  # [B, S]

        if DEBUG:
            for b in range(B):
                aw_list = [round(float(x), 6) for x in attn_weights[b].tolist()]
                ssum = round(float(attn_weights[b].sum().item()), 6)
                print(f"[BATCH {b}] attn_weights a_t (shape {tuple(attn_weights[b].shape)}): {aw_list}  (sum={ssum})")

        # 3) Weighted sum of encoder_outputs => context
        #    attn_weights => [batch_size, 1, src_len]
        #context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        context = torch.zeros(B, H)
        for b in range(B):
            c = torch.zeros(H)
            if DEBUG:
                print(f"[BATCH {b}] Computing context = Σ_i a_{b,i} * h_i:")

            for i in range(S):
                weight = attn_weights[b, i]
                h_i    = encoder_outputs[b, i]
                c      = c + weight * h_i

                if DEBUG:
                    w_val = round(float(weight.item()), 6)
                    h_list = [round(float(x), 4) for x in h_i.tolist()]
                    c_list = [round(float(x), 6) for x in c.tolist()]
                    print(f"  add a[{i}] * h_{i}:  {w_val} * {h_list[:4]}  -> partial context: {c_list[:4]}")

            context[b] = c

            if DEBUG:
                c_final = [round(float(x), 6) for x in context[b].tolist()]
                print(f"[BATCH {b}] FINAL context (shape {tuple(context[b].shape)}): {c_final[:4]}")

        return context, attn_weights


class EncoderCellRNN(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        """
        :param input_dim:  size of the source vocabulary
        :param embed_dim:  embedding dimension
        :param hidden_dim: hidden dimension of the RNN cell
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embedding for source tokens
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # RNN cell, Please use nn.GRUCell
        self.rnn_cell = nn.GRUCell(embed_dim,hidden_dim)

    def forward(self, src):
        """
        :param src: [batch_size, src_len] of token IDs
        :return:
          outputs: [batch_size, src_len, hidden_dim] (all hidden states)
          hidden:  [batch_size, hidden_dim]           (final hidden state)
        """
        batch_size, src_len = src.shape

        # Embed entire sequence
        embedded = self.embedding(src)  # => [batch_size, src_len, embed_dim]
        if DEBUG:
            print(f'embedded: {embedded.shape}')

        # Initialize hidden state to zeros
        hidden = torch.zeros(batch_size,self.hidden_dim)
        if DEBUG:
            print(f'hidden init: {hidden.shape}')
        all_outputs = []

        # Manually process each timestep
        for t in range(src_len):
            # embedded[:, t, :] => [batch_size, embed_dim] for step t
            x_t = embedded[:,t,:]
            hidden = self.rnn_cell(x_t,hidden)
            if DEBUG:
                print(f't={t} x_t: {x_t.shape}')
                print(f't={t} hidden: {hidden.shape}')
            # Add hidden state at step t to all_outputs here
            all_outputs.append(hidden.unsqueeze(1))

        # Combine all hidden states into a single tensor
        outputs = torch.cat(all_outputs, dim=1)  # => [batch_size, src_len, hidden_dim]
        # 1) Number of time steps collected
        if DEBUG:
            print("len(all_outputs):", len(all_outputs))  # expect 7

        # 2) Shape of each item
        if DEBUG:
            for i, t in enumerate(all_outputs):
                print(f"all_outputs[{i}] shape:", t.shape)  # each should be [2, 1, 512]

        # 3) Concatenate across time and check final shape
        outputs = torch.cat(all_outputs, dim=1)
        if DEBUG:
            print("outputs shape:", outputs.shape)         # expect [2, 7, 512]

        return outputs, hidden


class DecoderCellRNN(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, attention):
        """
        :param output_dim:  size of the target vocabulary
        :param embed_dim:   embedding dimension
        :param hidden_dim:  hidden dimension
        :param attention:   an attention module (e.g., DotAttention)
        """
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.attention = attention

        # Embedding for target tokens
        self.embedding = nn.Embedding(output_dim, embed_dim)

        # RNN cell that processes [embedded token + attention context], use another RNN cell for decoder.
        # Care input dimension. Concatenation of [embedded token + attention context]
        self.rnn_cell =nn.GRUCell(input_size=embed_dim + hidden_dim, hidden_size=hidden_dim)

        # Final projection to next-token logits
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, tgt, hidden, encoder_outputs):
        """
        :param tgt:            [batch_size, tgt_len] of target token IDs
        :param hidden:         [batch_size, hidden_dim] initial hidden state
        :param encoder_outputs:[batch_size, src_len, hidden_dim] from the encoder
        :return:
           logits:  [batch_size, tgt_len, output_dim]
           hidden:  [batch_size, hidden_dim] final decoder state
        """
        batch_size, tgt_len = tgt.shape

        # We'll store the predictions for each time step
        all_logits = torch.zeros(batch_size, tgt_len, self.output_dim)

        # Iterate over each timestep of the target sequence
        for t in range(tgt_len):

            # 1) embed the current target token
            tgt_t = tgt[:,t]
            emb_t = self.embedding(tgt_t)
            # 2) attention context from the encoder outputs. Call attention mechanism.
            context_t, attn_weights_t = self.attention(hidden, encoder_outputs)

            # 3) combine embedding + context and update hidden state
            rnn_input = torch.cat([emb_t, context_t], dim=-1)        # => [batch_size, embed_dim + hidden_dim]
            hidden = self.rnn_cell(rnn_input, hidden)                # => [batch_size, hidden_dim]

            # 4) project to next-token logits
            logits = self.fc_out(hidden)  # => [batch_size, output_dim]
            all_logits[:, t, :] = logits

        return all_logits, hidden

def main():
    if DEBUG:
        print("Debug mode on")
    # Hyperparameters (example)
    INPUT_DIM = 32     # source vocabulary size
    OUTPUT_DIM = 64    # target vocabulary size
    EMBED_DIM = 256
    HIDDEN_DIM = 512

    # Instantiate encoder, attention, and decoder
    encoder = EncoderCellRNN(INPUT_DIM, EMBED_DIM, HIDDEN_DIM)
    attention = DotAttention()
    decoder = DecoderCellRNN(OUTPUT_DIM, EMBED_DIM, HIDDEN_DIM, attention)

    # Example input data
    batch_size = 2
    src_len = 7
    tgt_len = 5
    src = torch.randint(0, INPUT_DIM, (batch_size, src_len))
    tgt = torch.randint(0, OUTPUT_DIM, (batch_size, tgt_len))

    # Forward pass
    encoder_outputs, encoder_final_hidden = encoder(src)
    # encoder_final_hidden => [batch_size, hidden_dim]

    # Decoder usually starts hidden state = encoder final (can vary)
    decoder_hidden = encoder_final_hidden

    # Decode the entire target sequence
    logits, final_hidden = decoder(tgt, decoder_hidden, encoder_outputs)

    print("Logits shape:", logits.shape)  # [batch_size, tgt_len, OUTPUT_DIM]

    # Your output should be torch.Size([2, 5, 64])

if __name__ == "__main__":
    DEBUG = False  #Change the debug value here to meet your needs
    main()