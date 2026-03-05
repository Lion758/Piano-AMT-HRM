embedding = torch.cat(
    (puzzle_embedding.view(-1, puzzle_emb_len, hidden_size), token_embedding), 
    dim=-2
)
```

For Sudoku and maze, no puzzle embeddings (every example is independently generated). For ARC, the embedding is crucial — different ARC tasks require completely different transformation rules, and the embedding encodes what the model has learned about this particular puzzle family from seeing its training examples.

The `CastedSparseEmbedding` uses a custom `SignSGD` optimizer because these embeddings are updated very sparsely — only embeddings for puzzles in the current batch change. Standard Adam accumulates stale moment estimates for rarely-seen embeddings, leading to poor updates.

---

### 1.7 Input/Output Representation

**Input:** Flat integer sequences. ARC: 30×30 grid → 900 integers (0=PAD, 1=EOS, 2–11=colors). Sudoku: 9×9 → 81 integers.

**Output:** Logits over the vocabulary at every sequence position, predicted simultaneously (non-autoregressive). The `lm_head` linear projects each z_H position to vocab_size logits.

**Loss:** `stablemax_cross_entropy` replaces softmax with the function `s(x) = 1/(1-x+ε)` for x<0, else `x+1`. This is numerically stable with large logits and avoids gradient vanishing near saturation.

---

## Part II: Sudoku Walkthrough — Step by Step

### Data Preparation

A 9×9 Sudoku board arrives as:
- `inputs`: shape `[B, 81]` — puzzle with zeros for blanks, shifted to range 1–10 (1=blank, 2–10=digits 1–9)
- `labels`: shape `[B, 81]` — complete solution in same encoding
- `puzzle_identifiers`: shape `[B]` — all zeros (Sudoku has no puzzle identity)

### Call 0: `model.initial_carry(batch)`

Creates the carry with `halted=True` for all B examples and empty, uninitialized z_H and z_L. The `halted=True` flag on the first call is what triggers the state reset.

### Calls 1–16: The Thinking Loop

**Call 1 — ACT Step 1:**

Because `halted=True` everywhere, all state is reset:
```
z_H[b] = H_init  (learned, broadcast to [81, 512])
z_L[b] = L_init  (learned, broadcast to [81, 512])
new_steps = 0 → 1
current_data = input batch

Spectrogram Diffusion (Hawthorne et al., 2022): A diffusion model that iteratively refines the piano roll given the audio features. This is conceptually the closest to HRM — iterative refinement from an initial estimate toward the target piano roll. It achieves strong results, suggesting the iterative refinement philosophy is genuinely beneficial for AMT.