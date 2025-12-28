# RNK

- RNK is a model idea that I've had for a while, and I finally got around to implementing it.
- It's a recursive neuro-symbolic model that processes text through a series of steps:
    1. Chunk encoding (tokens → latent chunks)
    2. State-space processing (fast local + slow long-term memory)
    3. Hierarchical reasoning (abstract pattern extraction)
    4. Neuro-symbolic refinement (constraint correction)
    5. Decoding (latent → tokens)
- The key insight: each chunk is "thought through" via HRM+NS
    before decoding, giving coherence through planning rather than
    pure autoregression.
- This is the best model I've made so far, and I'm excited to see where it goes. Currently, it's at it's peak until I add some 'theorized' features to it. (Well not really theorized since some has been proven, but adding these will change the model completely.)
- Also, it's not that good, but at only 5 Million parameters, the model can stay a tiny bit on par with the conversation:

```
Prompt: What color is the sky?
Response:  What is your favorite color that the wind? Sun is the sun? What

Prompt: Hello
Response:  Hello your young happiness as life abadain interons easily. life bother

Prompt: Hello, how are you?
Response:  Hello, how are you feeling? up? I are busy how emotions are you, how are you?? time are fun when you are?
```
- I mean now that I look at it, it's not that good at all, but cmon, what can you expect from a 5 Million parameter model?

- I'm also testing it on 50k samples from tinychat.jsonl dataset.