# Generative AI with LLM (Large Language Models)
- Generative AI and Large Language Models (LLMs) are general-purpose technologies, meaning they are not limited to a single application but can be utilized across a wide range of applications.

__First Week:__
- Examine transformers that power large language models
- Explore how these models are trained
- Understand the computing resources required to develop these powerful LLMs
- Learn about the technique called in-context learning
- Learn how to guide the model to output at inference time with prompt engineering
- Discover how to tune the most important generation parameters of LLMs to refine your model output
- Construct and compare different prompts and inputs for a given generative task, such as dialogue summarization
- Explore different inference parameters and sampling strategies to gain intuition on how to improve generative model responses further

__Second Week:__
- Explore options for adapting pre-trained models to a specific task and dataset via a process called instruction fine-tuning
- Learn how to align the output of language models with human values to increase helpfulness and decrease potential harm and toxicity
- Fine-tune an existing large language model from Hugging Face
- Experiment with both full fine-tuning and parameter-efficient fine-tuning (PEFT) and see how PEFT makes your workflow much more efficient

__Third Week:__
- Get hands-on with reinforcement learning from human feedback (RLHF)
- Build a reward model classifier to label model responses as either toxic or non-toxic

## Introduction:
- Attention Is All You Need: The concept of self-attention and multi-headed self-attention mechanisms are fundamental in the design of transformer models, allowing them to weigh the importance of different words in a sentence relative to each other.
- You can either use a pre-trained model or pre-train your own model. After that, you may choose to fine-tune and customize the model for your specific data to enhance its performance for your application.
- There are many large language models, some of which are open-source and others that are proprietary. It's important to choose the right model size for your needs. For instance, a giant model with 100 billion parameters may be necessary for comprehensive tasks, while a model with 1 to 30 billion parameters might suffice for more specific applications.
- In some cases, you need your model to be comprehensive and able to generalize across a wide range of tasks. Alternatively, there may be scenarios where optimizing for a single use case is sufficient, allowing you to achieve excellent results with a smaller model.
- Sometimes, small models can still provide significant capabilities and perform well in specific applications.
- When you want your language model to have extensive general knowledge about the world, such as understanding history, philosophy, and how to write Python code, a larger model with hundreds of billions of parameters can be beneficial. However, for tasks like summarizing dialogue or serving as a customer service agent for one company, using such a large model is often unnecessary, and a smaller model may be more practical.

## Generative AI & LLM:
- In this section, we will discuss large language models, their use cases, how they work, prompt engineering, generating creative text outputs, and outlining a project lifecycle for generative AI projects.
- Use Cases: Large language models can be used for creating chatbots, generating images from text, and using plugins to help develop code. These tools are capable of creating content that mimics human abilities.
- Generative AI is a subset of traditional machine learning. The models that underpin generative AI learn these abilities by finding statistical patterns in massive datasets of human-generated content.
- Large language models have been trained on trillions of words over a long period of time, using significant amounts of computational power.
- These foundation models, with billions of parameters, exhibit emergent properties beyond language alone. Researchers are unlocking their ability to break down complex tasks, reason, and solve problems.
- Below is a collection of foundation models, sometimes called base models, and their relative size in terms of parameters. Generally, the more parameters a model has, the more memory it requires and the more sophisticated tasks it can perform.

![image](https://github.com/user-attachments/assets/8b46967e-cb7a-455b-92d0-05133ae7168e)

- In the lab, you will use the open-source model Flan-T5. By either using these models as they are or applying fine-tuning techniques to adapt them to your specific use case, you can rapidly build customized solutions without the need to train a new model from scratch.
- In this course, you'll focus on large language models and their uses in natural language generation. You will learn how they are built and trained, how you can interact with them via text known as prompts, how to fine-tune models for your specific use case and data, and how to deploy them in applications to solve business and social tasks.
- In traditional machine learning, you write computer code with formalized syntax to interact with libraries and APIs. In contrast, large language models can understand and execute tasks from natural language or human-written instructions. The text you provide to an LLM is known as a prompt. The space or memory available for the prompt is called the context window, which is typically large enough for a few thousand words but differs from model to model.
- When a prompt is passed to the model, the model predicts the next words based on the input. If your prompt contains a question, the model generates an answer. The output of the model is called a completion, and the act of using the model to generate text is known as inference.
- The completion consists of the text from the original prompt followed by the generated text.

## LLM use cases and Tasks:
- As the scale of foundation models grows from hundreds of millions of parameters to billions, even hundreds of billions, the language understanding that a model possesses also increases. This understanding, stored within the parameters of the model, processes, reasons, and ultimately solves the tasks you give it.
- However, it's also true that smaller models can be fine-tuned to perform well on specific, focused tasks. These smaller models can be more efficient for specialized applications, where domain-specific knowledge or constraints are necessary.
- Next word prediction is the fundamental concept behind a number of different capabilities, such as text summarization, text translation, information retrieval systems, and named entity recognition.
- Large language models (LLMs) can be augmented by connecting them to external data sources or using them to invoke external APIs. This allows the model to access information that it doesn't have from its pre-training and enables it to power interactions with the real world.
- There are various use cases of LLMs such as Text Summarization, Text Translation, Information Retrieval, Named Entity Recognition (LLMs can identify and classify entities within text, such as people, organizations, locations, and more, which is essential for various applications in data extraction and analysis), Content Generation, Sentiment Analysis, Question Answering, Chatbots and Virtual Assistants, Code Generation and Assistance, Medical Diagnosis Assistance, Personalized e-learning and Tutoring

__Additional Use Cases:__
- Legal Document Analysis: LLMs can assist in analyzing legal documents, identifying key clauses, summarizing contracts, and ensuring compliance with legal standards.
- Financial Analysis and Forecasting: LLMs can analyze financial reports, market trends, and economic data to provide insights and forecasts, aiding in investment decisions and risk management.
- Customer Service Automation: LLMs can automate customer service tasks by handling inquiries, resolving issues, and providing personalized support, improving efficiency and user satisfaction.
- Social Media Monitoring: LLMs can track and analyze social media trends, sentiments, and mentions, helping brands manage their online presence and respond to public opinion.
- Product Recommendations: LLMs can power recommendation engines that suggest products or services to users based on their preferences and behavior, enhancing user experience and sales.
- Research and Development: LLMs can assist researchers by generating hypotheses, analyzing data, and identifying relevant literature, accelerating the pace of scientific discovery.
- Gaming and Interactive Storytelling: LLMs can create dynamic and interactive storytelling experiences in games, adapting narratives based on player choices and generating immersive content.

## Text generation before transformers:
- Generative algorithms are not new, previous generative algorithms use Recurrent neural networks (RNN).
- Learn RNN
- Homonyms
- cons of RNN
- Ambiguity
- After RNN attention is all you need to publish and transformers are introduced and everything is changed.
- It can be scaled efficiently to use multi-core GPUs
- It can parallel process input data making use of a much larger training dataset
- It is able to pay attention to the meaning of the word processing.

## Transformer architecture:
- Building large language models using the transformer architecture dramatically improved the performance of natural language tasks over the earlier generation of RNNs, and led to an explosion in regenerative capability.
- The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a sentence. To apply attention weights to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input.


![image](https://github.com/user-attachments/assets/57a07c9a-abc1-4cc6-9de5-701b36e9c434)

- Learn self-attention model

![image](https://github.com/user-attachments/assets/175f0531-fbaf-4a5b-8ef4-212b964e2c9f)

## Simplifying transformer architecture
- The transformer architecture is split into two distinct parts Encoder and decoder.
- These components work in conjunction with each other and they share a number of similarities.
- Machine learning models are just big statistical calculators and they work with numbers, not words. So before passing the text into the model process you first tokenize the words. This converts words into numbers representing a position in a dictionary of all the possible words that the model can work with.
- What are the various tokenization methods?
- Once you select a tokenizer to train the model you must use the same tokenizer when you generate text.
- After tokenization pass it to the embedding layer (This is a trainable vector embedding space) high dimensional space where each token is represented as a vector and occupies a unique location within that space.
- Each token ID in the vocabulary is matched to a multi-dimensional vector and the intuition is that these vectors learn to encode the meaning and context of individual tokens in the input sequence.
- Embedding vector sequences have been using natural language processing for some time previous generation language algorithms like word2vec use this concept.
- Each word is matched to a token ID and each token ID is mapped to a vector.
- In the embedding space you can calculate the distance between the words as an angle 
- As you add the token vectors into the base encoder or the decoder you also ass positional encoding, the model processes each of the input tokens in parallel. So by adding the positional encoding you preserve the information about the word order and don't lose the relevance of the position of the word in a sentence. 
- Once you summed the input tokens and the positional encoding you pass the resulting vector to the self-attention layer.
- Here the model analyzes the relationships between the tokens in your input sequence. This allows the model to attend two different  parts of the input sequence to better capture the contextual dependencies between the words.
- The self-attention weights that are learned during training and stored in these layers reflect the importance of each word in that input sequence to all other words in the sequence. But this does not happen just once the transformer architecture actually has multi-headed self-attention this means the multiple sets of self-attention weights or heads are learned in parallel and independently of each other.
- The number of attention heads included in the attention layer varies from model to model but numbers in the range of 12-100 are common.
- The intuition here is that each delf attention head will learn a different aspect of language.
- For example, one head may see the relationship between the people entities in our sentence. While another head may focus on the activity of the sentence. Whilst yet another head may focus on some other properties such as if the words rhyme.
- The weights of each head are randomly initialized and given sufficient training data and time, each will learn different aspects of language.
- Now that all of the attention weights have been applied to your input data, the output is processed through a fully connected feed-forward network. The output of this layer is a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary.
- You can then pass these logits to a final softmax layer, where they are normalized into a probability score for each word. This output includes a probability for every single word in the vocabulary, so there's likely to be thousands of scores here.
- One single token will have a score higher than the rest. This is the most likely predicted token.

## Generating text with transformers:
- What is the seq-to-seq model
- Translation task using transformer:
  - Tokenization: Tokenize the input words using the same tokenizer which is used to train the network.
  - Tokens are then added to the input on the encoder side of the network
  - passed through the word embedding layer and then fed to multi multi-headed attention layer 
  - The output of multi-headed attention layers is fed through a feed-forward network to the output of the encoder
  - Data that leaves the encoder is a deep representation of the structure and meaning of the input sequence.
  - This representation is inserted into the middle of the decoder to influence the decoder's self-attention mechanism
  The start of sequence token is added to the input of the decoder this triggers the decoder to predict the next token based on the contextual understanding that it's being provided by the encoder
  - The output of the decoder's self-attention layers gets passed through the feed-forward network and through a final softmax output layer.
  - At this point, we will have our first token we will continue this loop passing the output token back to the input to trigger the generation of the next token until the model predicts the end of the sequence token
  - The final sequence of the tokens can be tokenized into words and we will have our output
- There are multiple ways in which you can use the output from the sotmax layer to predict the next token which can influence how creative your generated text is.

## Summerize:
- The complete transformer architecture consists of an encoder and decoder components. The encoder encodes input sequences into a deep representation of the structure and meaning of the input. The decoder, working from input token triggers, uses the encoder's contextual understanding to generate new tokens. It does this in a loop until some stop condition has been reached.

## Variations of the architecture:
- Encoder only: work as sequence-to-sequence models, but without further modification, the input sequence and the output sequence or the same length. BERT is an example of an encoder-only model perform classification tasks such as sentiment analysis
- Encoder-Decoder models Encoder-decoder models, as you've seen, perform well on sequence-to-sequence tasks such as translation, where the input sequence and the output sequence can be different lengths. BART, T5
- Decoder only models: These models can now generalize to most tasks. Popular decoder-only models include the GPT family of models, BLOOM, Jurassic, LLaMA, and many more.

## Prompting and Prompt engineering:
- The text that you feed into the model is called the prompt, the act of generating text is known as inference, and the output text is known as the completion.The full amount of text or the memory that is available to use for the prompt is called the context window.
- you'll frequently encounter situations where the model doesn't produce the outcome that you want on the first try. You may have to revise the language in your prompt or the way that it's written several times to get the model to behave in the way that you want. This work to develop and improve the prompt is known as prompt engineering.
- In context learning: But one powerful strategy to get the model to produce better outcomes is to include examples of the task that you want the model to carry out inside the prompt. Providing examples inside the context window is called in-context learning.
- With in-context learning, you can help LLMs learn more about the task being asked by including examples or additional data in the prompt.
- For example: Classify this review: "I love this movie" Sentiment?
- including your input data within the prompt, is called zero-shot inference.  
- Providing the examples within the prompt:For example: Classfiy this review: "I loved this movie" Sentiment: Positive, Classify this review: "I don't like this movie" Sentiment:?
- Inclusion of single example is known as one shot inferance.
- Sometimes one single example wont be enough for the model to learn what yoy want it to do So you can extend the idea of giving a single example to include multiple examples. This is known as few-shot inference.
-  While the largest models are good at zero-shot inference with no examples, smaller models can benefit from one-shot or few-shot inference that include examples of the desired behavior. But remember the context window because you have a limit on the amount of in-context learning that you can pass into the model.
-  if you find that your model isn't performing well when, say, including five or six examples, you should try fine-tuning your model instead. Fine-tuning performs additional training on the model using new data to make it more capable of the task you want it to perform.

## Generative configuration: the methods and associated configuration parameters that you can use to influence the way that the model makes the final decision about next-word generation. 
- Each model exposes a set of configuration parameters that can influence the model's output during inference. these are different than the training parameters which are learned during training time.
- these configuration parameters are invoked at inference time and give you control over things like the maximum number of tokens in the completion, and how creative the output is.
- Max new tokens is probably the simplest of these parameters, and you can use it to limit the number of tokens that the model will generate.
- But note how the length of the completion in the example for 200 is shorter. This is because another stop condition was reached, such as the model predicting and end of sequence token. Remember it's max new tokens, not a hard number of new tokens generated.

![image](https://github.com/user-attachments/assets/05115eaf-cd27-4366-8bf0-097c52c92674)

- The output from the transformer's softmax layer is a probability distribution across the entire dictionary of words that the model uses.
- Most large language models by default will operate with so-called greedy decoding. This is the simplest form of next-word prediction, where the model will always choose the word with the highest probability.
- This method can work very well for short generation but is susceptible to repeated words or repeated sequences of words. If you want to generate text that's more natural, more creative and avoids repeating words, you need to use some other controls.
- Random sampling is the easiest way to introduce some variability. Instead of selecting the most probable word every time with random sampling, the model chooses an output word at random using the probability distribution to weight the selection.
- For example, in the illustration, the word banana has a probability score of 0.02. With random sampling, this equates to a 2% chance that this word will be selected. By using this sampling technique, we reduce the likelihood that words will be repeated. However, depending on the setting, there is a possibility that the output may be too creative, producing words that cause the generation to wander off into topics or words that just don't make sense.
- Note that in some implementations, you may need to disable greedy and enable random sampling explicitly.
- top k and top p sampling techniques: Two Settings, top p and top k are sampling techniques that we can use to help limit the random sampling and increase the chance that the output will be sensible.
- To limit the options while still allowing some variability, you can specify a top k value which instructs the model to choose from only the k tokens with the highest probability.
- This method can help the model have some randomness while preventing the selection of highly improbable completion words. This in turn makes your text generation more likely to sound reasonable and to make sense.
- Alternatively, you can use the top p setting to limit the random sampling to the predictions whose combined probabilities do not exceed p.
-  For example, if you set p to equal 0.3, the options are cake and donut since their probabilities of 0.2 and 0.1 add up to 0.3.
-  The model then uses the random probability weighting method to choose from these tokens. With top k, you specify the number of tokens to randomly choose from, and with top p, you specify the total probability that you want the model to choose from.
-  One more parameter that you can use to control the randomness of the model output is known as temperature. This parameter influences the shape of the probability distribution that the model calculates for the next token. Broadly speaking, the higher the temperature, the higher the randomness, and the lower the temperature, the lower the randomness.
-  The temperature value is a scaling factor that's applied within the final softmax layer of the model that impacts the shape of the probability distribution of the next token
-  In contrast to the top k and top p parameters, changing the temperature actually alters the predictions that the model will make.
-  If you choose a low value of temperature, say less than one, the resulting probability distribution from the softmax layer is more strongly peaked with the probability being concentrated in a smaller number of words.
-  The model will select from this distribution using random sampling and the resulting text will be less random and will more closely follow the most likely word sequences that the model learned during training.
- If instead you set the temperature to a higher value, say, greater than one, then the model will calculate a broader flatter probability distribution for the next token.
- This leads the model to generate text with a higher degree of randomness and more variability in the output compared to a cool temperature setting. This can help you generate text that sounds more creative.
- If you leave the temperature value equal to one, this will leave the softmax function as default and the unaltered probability distribution will be used.

## GEnerative AI project lifecycle:

![image](https://github.com/user-attachments/assets/f922bb62-21f3-47e6-a638-e4da8e5ce402)

- LLMs are capable of carrying out many tasks, but their abilities depend strongly on the size and architecture of the model. You should think about what function the LLM will have in your specific application. Do you need the model to be able to carry out many different tasks, including long-form text generation or with a high degree of capability, or is the task much more specific like named entity recognition so that your model only needs to be good at one thing
- Once you're happy, and you've scoped your model requirements enough to begin development. Your first decision will be whether to train your own model from scratch or work with an existing base model.
- In general, you'll start with an existing model, although there are some cases where you may find it necessary to train a model from scratch.
- Considerations as well as some rules of thumb to help you estimate the feasibility of training your own model behind this decision????
- With your model in hand, the next step is to assess its performance and carry out additional training if needed for your application.
- prompt engineering can sometimes be enough to get your model to perform well, so you'll likely start by trying in-context learning, using examples suited to your task and use case.
- There are still cases, however, where the model may not perform as well as you need, even with one or a few short inference, and in that case, you can try fine-tuning your model.
- . As models become more capable, it's becoming increasingly important to ensure that they behave well and in a way that is aligned with human preferences in deployment. reinforcement learning with human feedback, which can help to make sure that your model behaves well.
- An important aspect of all of these techniques is evaluation.  determine how well your model is performing or how well aligned it is to your preferences. this adapt and aligned stage of app development can be highly iterative.
-  You may start by trying prompt engineering and evaluating the outputs, then using fine tuning to improve performance and then revisiting and evaluating prompt engineering one more time to get the performance that you need. 
- when you've got a model that is meeting your performance needs and is well aligned, you can deploy it into your infrastructure and integrate it with your application.
- At this stage, an important step is to optimize your model for deployment. This can ensure that you're making the best use of your compute resources and providing the best possible experience for the users of your application.
- The last but very important step is to consider any additional infrastructure that your application will require to work well. There are some fundamental limitations of LLMs that can be difficult to overcome through training alone like their tendency to invent information when they don't know an answer, or their limited ability to carry out complex reasoning and mathematics.

## Introduction to AWS:
- load the flan T5 model
- load the tokenizer (Autotokenizer converts raw text to a vector space)
- Summarize data with instruction tuning (in context learning specifically in context learning with zero shot instruction)
- Use different prompt
- Try finding best prompt
- Try one shot, few shot inferance  
- if you have no idea how a model is, if you just get it off of some model hub somewhere. These are the first step. Prompt engineering, zero-shot, one-shot, few shot is almost always the first step when you're trying to learn the language model that you've been handed and dataset.
- In a case if it is exiding the context window then you need to filter out those inputs
- people often try to just keep adding more and more shots, five shots, six shots. Typically, in my experience, above five or six shots, so full prompt and then completions, you really don't gain much after that. Either the model can do it or it can't do it and going about five or six.
- Try config parameters (top k, top p, temp)

## Pre-training large language models:
- There are specific circumstances where training your own model from scratch might be advantageous
- The developers of some of the major frameworks for building generative AI applications like Hugging Face and PyTorch, have curated hubs where you can browse these models.
- A really useful feature of these hubs is the inclusion of model cards, that describe important details including the best use cases for each model, how it was trained, and known limitations.

![image](https://github.com/user-attachments/assets/b83c57ff-2dc8-4d00-9e65-0782ee81bf24)

- Variance of the transformer model architecture are suited to different language tasks, largely because of differences in how the models are trained.

## High level of the initial training process of LLMs(pre-training)
- LLMs encode a deep statistical representation of language. This understanding is developed during the model's pre-training phase when the model learns from vast amounts of unstructured textual data.
- This can be gigabytes, terabytes, and even petabytes of text. This data is pulled from many sources, including scrapes off the Internet and corpora of texts that have been assembled specifically for training language models.
- During pre-training, the model weights get updated to minimize the loss of the training objective. The encoder generates an embedding or vector representation for each token. Pre-training also requires a large amount of compute and the use of GPUs.
- when you scrape training data from public sites such as the Internet, you often need to process the data to increase quality, address bias, and remove other harmful content.
- Encoder-only models are also known as Autoencoding models, and they are pre-trained using masked language modeling. Here, tokens in the input sequence or randomly mask, and the training objective is to predict the mask tokens in order to reconstruct the original sentence. This is also called a denoising objective. Autoencoding models spilled bi-directional representations of the input sequence, meaning that the model has an understanding of the full context of a token and not just of the words that come before. Encoder-only models are ideally suited to task that benefit from this bi-directional contexts. You can use them to carry out sentence classification tasks, for example, sentiment analysis or token-level tasks like named entity recognition or word classification. Some well-known examples of an autoencoder model are BERT and RoBERTa.
- Decoder-only models are also known as Autoregressive models. which are pre-trained using causal language modeling. Here, the training objective is to predict the next token based on the previous sequence of tokens. Predicting the next token is sometimes called full language modeling by researchers. Decoder-based autoregressive models, mask the input sequence and can only see the input tokens leading up to the token in question. The model has no knowledge of the end of the sentence. The model then iterates over the input sequence one by one to predict the following token. In contrast to the encoder architecture, this means that the context is unidirectional. By learning to predict the next token from a vast number of examples, the model builds up a statistical representation of language. Models of this type make use of the decoder component off the original architecture without the encoder. Decoder-only models are often used for text generation, although larger decoder-only models show strong zero-shot inference abilities, and can often perform a range of tasks well. Well known examples of decoder-based autoregressive models are GBT and BLOOM.
- Encoder-Decoder models are also known as sequence-to-sequence models.  that uses both the encoder and decoder parts off the original transformer architecture. The exact details of the pre-training objective vary from model to model. A popular sequence-to-sequence model T5, pre-trains the encoder using span corruption, which masks random sequences of input tokens. Those mass sequences are then replaced with a unique Sentinel token, shown here as x. Sentinel tokens are special tokens added to the vocabulary, but do not correspond to any actual word from the input text. The decoder is then tasked with reconstructing the mask token sequences auto-regressively. The output is the Sentinel token followed by the predicted tokens. You can use sequence-to-sequence models for translation, summarization, and question-answering. They are generally useful in cases where you have a body of texts as both input and output.another well-known encoder-decoder model is BART.

![image](https://github.com/user-attachments/assets/33cb2a35-b0e7-4b0d-8add-913d9f7a9ca4)

![image](https://github.com/user-attachments/assets/cb493711-c683-4c43-a5e0-8c9679007fed)

- Autoencoding models are pre-trained using masked language modeling. They correspond to the encoder part of the original transformer architecture, and are often used with sentence classification or token classification.
- Autoregressive models are pre-trained using causal language modeling. Models of this type make use of the decoder component of the original transformer architecture, and often used for text generation.
- equence-to-sequence models use both the encoder and decoder part off the original transformer architecture. The exact details of the pre-training objective vary from model to model. The T5 model is pre-trained using span corruption. Sequence-to-sequence models are often used for translation, summarization, and question-answering.

## Challenges in training LLMs:
- One common issue encountered while training large language models is the OutOfMemoryError, which occurs when CUDA, a collection of libraries and tools developed for Nvidia GPUs and used by frameworks like PyTorch and TensorFlow to accelerate deep learning operations, runs out of memory
- LLMs require a significant amount of memory to store their parameters, with each parameter typically needing 4 bytes (32-bit float), meaning 1 billion parameters require 4 GB of RAM. However, to train the model, you need additional memory for 2 ADAM optimizers, gradients, activations, and temporary memory used by your function. In total, you'll actually need approximately 6 times the amount of GPU RAM that the model weights alone require.
- To train a one billion parameter model at 32-bit full precision, you'll need approximately 24 gigabyte of GPU RAM.
- By default, model weights, activations, and other model parameters are stored in FP32.
- Quantization statistically projects the original 32-bit floating point numbers into a lower precision space, using scaling factors calculated based on the range of the original 32-bit floating point numbers.
- you lose some precision with this projection.this loss in precision is acceptable in most cases because you're trying to optimize for memory footprint.
- Storing a value in FP32 requires four bytes of memory. In contrast, storing a value on FP16 requires only two bytes of memory, so with quantization you have reduced the memory requirement by half.
- One datatype in particular BFLOAT16, has recently become a popular alternative to FP16. BFLOAT16, short for Brain Floating Point Format developed at Google Brain has become a popular choice in deep learning.
- Many LLMs, including FLAN-T5, have been pre-trained with BFLOAT16. BFLOAT16 or BF16 is a hybrid between half precision FP16 and full precision FP32.
- BF16 significantly helps with training stability and is supported by newer GPU's such as NVIDIA's A100. BFLOAT16 is often described as a truncated 32-bit float, as it captures the full dynamic range of the full 32-bit float, that uses only 16-bits.
- This not only saves memory, but also increases model performance by speeding up calculations. The downside is that BF16 is not well suited for integer calculations,
- the goal of quantization is to reduce the memory required to store and train models by reducing the precision off the model weights.
- Quantization statistically projects the original 32-bit floating point numbers into lower precision spaces using scaling factors calculated based on the range of the original 32-bit floats.
- BFLOAT16 has become a popular choice of precision in deep learning as it maintains the dynamic range of FP32, but reduces the memory footprint by half. Many LLMs, including FLAN-T5, have been pre-trained with BFOLAT16.
- By applying quantization, you can reduce your memory consumption required to store the model parameters down to only two gigabyte using 16-bit half precision of 50% saving and you could further reduce the memory footprint by another 50% by representing the model parameters as eight bit integers, which requires only one gigabyte of GPU RAM.
- As modal scale beyond a few billion parameters, it becomes impossible to train them on a single GPU. Instead, you'll need to turn to distributed computing techniques while you train your model across multiple GPUs. which is very expensive. 


## Model sharding: 
Fully Sharded Data Parallel (FSDP) in PyTorch is a popular implementation that builds on the Zero Redundancy Optimizer (ZeRO) technique. ZeRO was designed to optimize memory usage by distributing or "sharding" model states—such as parameters, gradients, and optimizer states—across multiple GPUs. This approach is particularly beneficial when the model size exceeds the memory capacity of a single GPU, allowing the model to be trained in a distributed manner across several GPUs without running into memory limitations.

How ZeRO Works
ZeRO addresses a key limitation in Distributed Data Parallel (DDP) training, where each GPU maintains a full copy of the model, leading to high memory consumption. By contrast, ZeRO shards the model parameters, gradients, and optimizer states across GPUs, significantly reducing the memory footprint.

ZeRO Stage 1 shards only the optimizer states across GPUs, which can reduce memory usage by up to a factor of four.
ZeRO Stage 2 extends this by also sharding the gradients, achieving up to an eightfold reduction in memory usage when combined with Stage 1.
ZeRO Stage 3 takes this further by sharding the model parameters in addition to the optimizer states and gradients. When all three stages are used together, the memory reduction is proportional to the number of GPUs, meaning a setup with 64 GPUs could reduce the memory usage by 64 times.
FSDP vs. DDP
Unlike DDP, where each GPU has a full copy of all the model states needed for processing, FSDP requires GPUs to communicate with each other to collect the necessary sharded data for the forward and backward passes. This on-demand collection and release of data introduce a trade-off between performance and memory usage. While FSDP's approach reduces overall GPU memory utilization, it increases communication overhead, which can affect performance, especially as the model size grows and is distributed across more GPUs.

FSDP also offers flexibility in managing this trade-off by allowing you to configure the "sharding factor." A sharding factor of one results in no sharding (similar to DDP), while the maximum sharding factor fully shards the model across all available GPUs, providing the greatest memory savings but at the cost of increased inter-GPU communication.

Handling Large Models
When training very large models—such as those with billions of parameters—DDP often encounters out-of-memory errors because it replicates the entire model on each GPU. FSDP, however, can efficiently handle such large models by distributing the model states across GPUs. Additionally, using techniques like lowering precision (e.g., to 16-bit) can further improve computational efficiency, leading to higher teraflops.

Communication Overhead
One of the trade-offs with FSDP is the increased communication overhead as the model size grows and the number of GPUs involved in training increases. This communication overhead can slow down training as more data needs to be exchanged between GPUs, which becomes a bottleneck, especially in large-scale training setups.


## Relationship between model size, training confga and performance to determine how big the model is:
- The goal during pre-training is to maximize the model's performance of its learning objective, which is minimizing the loss when predicting tokens.
- Two options you have to achieve better performance are increasing the size of the dataset you train your model on and increasing the number of parameters in your model. You could scale either of both of these quantities to improve performance.
- another issue to take into consideration is your compute budget which includes factors like the number of GPUs you have access to and the time you have available for training models.

![image](https://github.com/user-attachments/assets/188d91c0-4e92-4551-884e-541d90ee82cc)

## Unit of compute that quantifies the required resources:
- A petaFLOP per second day is a measurement of the number of floating point operations performed at a rate of one petaFLOP per second, running for an entire day. One petaFLOP corresponds to one quadrillion floating point operations per second.
- larger numbers can be achieved by either using more compute power or training for longer or both.
- A power law is a mathematical relationship between two variables, where one is proportional to the other raised to some power.
- this would suggest that you can just increase your compute budget to achieve better model performance. In practice however, the compute resources you have available for training will generally be a hard constraint set by factors such as the hardware you have access to, the time available for training and the financial budget of the project. If you hold your compute budget fixed, the two levers you have to improve your model's performance are the size of the training dataset and the number of parameters in your model.
- The OpenAI researchers found that these two quantities also show a power-law relationship with a test loss in the case where the other two variables are held fixed.

![image](https://github.com/user-attachments/assets/4cf64e79-1768-4974-a8f7-91f059fa240c)

![image](https://github.com/user-attachments/assets/78cbb169-ee20-4340-8e41-31bb69fd9165)

# WEEK 2
Pretrained Base Models:

A pretrained base model contains valuable knowledge but may not respond effectively to prompts.
Instruction fine-tuning modifies the model's behavior to better handle specific queries.
Fine-Tuning Techniques:

LoRA (Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning) are methods to improve a model’s performance with fewer resources.
These techniques allow fine-tuning without updating all model parameters, making the process more efficient.
Improving Model Performance:

You can enhance a model's performance for a specific use case using fine-tuning methods.
Evaluating performance metrics is crucial to quantify the improvement over the base model.
Challenges with Zero- or Few-Shot Learning:

Small models often struggle with zero- or few-shot inference, even when multiple examples are provided.
Including examples in the prompt consumes valuable context window space, making fine-tuning a more effective approach.
Fine-Tuning vs. Pre-Training:

Pre-training involves training on vast amounts of unstructured data via self-supervised learning.
Fine-tuning is a supervised learning process using labeled prompt-completion pairs to update the model’s weights for specific tasks.
Instruction Fine-Tuning:

This process improves the model’s ability to handle a variety of tasks by using examples that show how to respond to specific instructions.
The training dataset includes many pairs of prompt-completion examples, each with an instruction.
Full Fine-Tuning:

In full fine-tuning, all of the model’s weights are updated, similar to pre-training.
This requires a substantial memory and compute budget to manage the gradients, optimizers, and other components involved in the training process.

### How do you actually go about instruction, fine-tuning and LLM? 
- The first step is to prepare your training data. There are many publicly available datasets that have been used to train earlier generations of language models, although most of them are not formatted as instructions. 
- prompt template libraries can be used to take existing datasets, and turn them into instruction prompt datasets for fine-tuning.
- Prompt template libraries include many templates for different tasks and different data sets.  The result is a prompt that now contains both an instruction and the example from the data set. 
- Once you have your instruction data set ready, as with standard supervised learning, you divide the data set into training validation and test splits. During fine tuning, you select prompts from your training data set and pass them to the LLM, which then generates completions. Next, you compare the LLM completion with the response specified in the training data.
- the output of an LLM is a probability distribution across tokens. So you can compare the distribution of the completion and that of the training label and use the standard crossentropy function to calculate loss between the two token distributions. And then use the calculated loss to update your model weights in standard backpropagation.
- there is a potential downside to fine-tuning on a single task. The process may lead to a phenomenon called catastrophic forgetting. Catastrophic forgetting happens because the full fine-tuning process modifies the weights of the original LLM. While this leads to great performance on the single fine-tuning task, it can degrade performance on other tasks.

to avoid catastrophic forgetting? First of all, it's important to decide whether catastrophic forgetting actually impacts your use case. If all you need is reliable performance on the single task you fine-tuned on, it may not be an issue that the model can't generalize to other tasks.
If you do want or need the model to maintain its multitask generalized capabilities, you can perform fine-tuning on multiple tasks at one time.

Good multitask fine-tuning may require 50-100,000 examples across many tasks, and so will require more data and compute to train.
Our second option is to perform parameter efficient fine-tuning, or PEFT for short instead of full fine-tuning. PEFT is a set of techniques that preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters.  PEFT shows greater robustness to catastrophic forgetting since most of the pre-trained weights are left unchanged.


Which of the following are true in respect to Catastrophic Forgetting? Select all that apply.


Catastrophic forgetting occurs when a machine learning model forgets previously learned information as it learns new information.

Correct
The assertion is true, and this process is especially problematic in sequential learning scenarios where the model is trained on multiple tasks over time.


One way to mitigate catastrophic forgetting is by using regularization techniques to limit the amount of change that can be made to the weights of the model during training.

Correct
One way to mitigate catastrophic forgetting is by using regularization techniques to limit the amount of change that can be made to the weights of the model during training. This can help to preserve the information learned during earlier training phases and prevent overfitting to the new data.


Catastrophic forgetting only occurs in supervised learning tasks and is not a problem in unsupervised learning.

This should not be selected
Catastrophic forgetting is a problem in both supervised and unsupervised learning tasks. In unsupervised learning, it can occur when the model is trained on a new dataset that is different from the one used during pre-training.


Catastrophic forgetting is a common problem in machine learning, especially in deep learning models.

Catastrophic forgetting occurs when a machine learning model forgets previously learned information as it learns new information. The assertion is true, and this process is especially problematic in sequential learning scenarios where the model is trained on multiple tasks over time.

One way to mitigate catastrophic forgetting is by using regularization techniques to limit the amount of change that can be made to the weights of the model during training.

One way to mitigate catastrophic forgetting is by using regularization techniques to limit the amount of change that can be made to the weights of the model during training.
One way to mitigate catastrophic forgetting is by using regularization techniques to limit the amount of change that can be made to the weights of the model during training. This can help to preserve the information learned during earlier training phases and prevent overfitting to the new data.

Catastrophic forgetting is a common problem in machine learning, especially in deep learning models. This assertion is true because these models typically have many parameters, which can lead to overfitting and make it more difficult to retain previously learned information.

## Multi-task instruction fine tuning:
- Multitask fine-tuning is an extension of single task fine-tuning, where the training dataset is comprised of example inputs and outputs for multiple tasks.
- Here, the dataset contains examples that instruct the model to carry out a variety of tasks, including summarization, review rating, code translation, and entity recognition. You train the model on this mixed dataset so that it can improve the performance of the model on all the tasks simultaneously, thus avoiding the issue of catastrophic forgetting
- Over many epochs of training, the calculated losses across examples are used to update the weights of the model, resulting in an instruction tuned model that is learned how to be good at many different tasks simultaneously. One drawback to multitask fine-tuning is that it requires a lot of data.
- Instruct model variance differ based on the datasets and tasks used during fine-tuning. One example is the FLAN family of models. FLAN, which stands for fine-tuned language net, is a specific set of instructions used to fine-tune different models.
- FLAN-T5, the FLAN instruct version of the T5 foundation model while FLAN-PALM is the flattening struct version of the palm foundation model.FLAN-T5 is a great general purpose instruct model. In total, it's been fine tuned on 473 datasets across 146 task categories. 
![image](https://github.com/user-attachments/assets/dd78a2cc-7f61-4c9f-b5a0-c88e61a43fb4)

One example of a prompt dataset used for summarization tasks in FLAN-T5 is SAMSum. It's part of the muffin collection of tasks and datasets and is used to train language models to summarize dialogue.
Including different ways of saying the same instruction helps the model generalize and perform better. Just like the prompt templates you saw earlier.

For example, imagine you're a data scientist building an app to support your customer service team, process requests received through a chat bot, like the one shown here. Your customer service team needs a summary of every dialogue to identify the key actions that the customer is requesting and to determine what actions should be taken in response. The SAMSum dataset gives FLAN-T5 some abilities to summarize conversations. However, the examples in the dataset are mostly conversations between friends about day-to-day activities and don't overlap much with the language structure observed in customer service chats. You can perform additional fine-tuning of the FLAN-T5 model using a dialogue dataset that is much closer to the conversations that happened with your bot. This is the exact scenario that you'll explore in the lab this week. You'll make use of an additional domain specific summarization dataset called dialogsum to improve FLAN-T5's is ability to summarize support chat conversations. This dataset consists of over 13,000 support chat dialogues and summaries. The dialogue some dataset is not part of the FLAN-T5 training data, so the model has not seen these conversations before. Let's take a look at example from dialogsum and discuss how a further round of fine-tuning can improve the model. This is a support chat that is typical of the examples in the dialogsum dataset. 
One thing you need to think about when fine-tuning is how to evaluate the quality of your models completions. 

In a case of LLM the output is non-deterministic and language-based evaluation is much more challenging.
ROUGE and BLEU, are two widely used evaluation metrics for different tasks. 
ROUGE or recall oriented under study for jesting evaluation is primarily employed to assess the quality of automatically generated summaries by comparing them to human-generated reference summaries. On the other hand, BLEU, or bilingual evaluation understudy is an algorithm designed to evaluate the quality of machine-translated text, again, by comparing it to human-generated translations.
In the anatomy of language, a unigram is equivalent to a single word. A bigram is two words and n-gram is a group of n-words.

ROUGE-1 metric
- It is cold outside and a generated output that is very cold outside. You can perform simple metric calculations similar to other machine-learning tasks using recall, precision, and F1.
- The recall metric measures the number of words or unigrams that are matched between the reference and the generated output divided by the number of words or unigrams in the reference.
- Precision measures the unigram matches divided by the output size. The F1 score is the harmonic mean of both of these values. 
These are very basic metrics that only focused on individual words, hence the one in the name, and don't consider the ordering of the words. It can be deceptive. It's easily possible to generate sentences that score well but would be subjectively poor.
Stop for a moment and imagine that the sentence generated by the model was different by just one word. Not, so it is not cold outside. The scores would be the same.

By using bigrams, you're able to calculate a ROUGE-2. Now, you can calculate the recall, precision, and F1 score using bigram matches instead of individual words. With longer sentences, they're a greater chance that bigrams don't match, and the scores may be even lower. Rather than continue on with ROUGE numbers growing bigger to n-grams of three or fours, let's take a different approach. 

Instead, you'll look for the longest common subsequence present in both the generated output and the reference output. You can now use the LCS value to calculate the recall precision and F1 score, where the numerator in both the recall and precision calculations is the length of the longest common subsequence, in this case, two. Collectively, these three quantities are known as the Rouge-L score.  The Rouge-1 precision score will be perfect. One way you can counter this issue is by using a clipping function to limit the number of unigram matches to the maximum count for that unigram within the reference.

The other score that can be useful in evaluating the performance of your model is the BLEU score, which stands for bilingual evaluation under study. Just to remind you that BLEU score is useful for evaluating the quality of machine-translated text. The score itself is calculated using the average precision over multiple n-gram sizes
Just like the Rouge-1 score that we looked at before, but calculated for a range of n-gram sizes and then averaged. Let's take a closer look at what this measures and how it's calculated.
The BLEU score quantifies the quality of a translation by checking how many n-grams in the machine-generated translation match those in the reference translation. To calculate the score, you average precision across a range of different n-gram sizes. You can use them for simple reference as you iterate over your models, but you shouldn't use them alone to report the final evaluation of a large language model. Use rouge for diagnostic evaluation of summarization tasks and BLEU for translation tasks.

LLMs are complex, and simple evaluation metrics like the rouge and blur scores, can only tell you so much about the capabilities of your model. In order to measure and compare LLMs more holistically, you can make use of pre-existing datasets, and associated benchmarks that have been established by LLM researchers specifically for this purpose.
Selecting the right evaluation dataset is vital, so that you can accurately assess an LLM's performance, and understand its true capabilities. You'll find it useful to select datasets that isolate specific model skills, like reasoning or common sense knowledge, and those that focus on potential risks, such as disinformation or copyright infringement. 
Benchmarks, such as GLUE, SuperGLUE, or Helm, cover a wide range of tasks and scenarios. They do this by designing or collecting datasets that test specific aspects of an LLM. GLUE, or General Language Understanding Evaluation, was introduced in 2018. GLUE is a collection of natural language tasks, such as sentiment analysis and question-answering. GLUE was created to encourage the development of models that can generalize across multiple tasks, and you can use the benchmark to measure and compare the model performance. As a successor to GLUE, SuperGLUE was introduced in 2019, to address limitations in its predecessor. It consists of a series of tasks, some of which are not included in GLUE, and some of which are more challenging versions of the same tasks. SuperGLUE includes tasks such as multi-sentence reasoning, and reading comprehension. Both the GLUE and SuperGLUE benchmarks have leaderboards that can be used to compare and contrast evaluated models. The results page is another great resource for tracking the progress of LLMs. As models get larger, their performance against benchmarks such as SuperGLUE start to match human ability on specific tasks. That's to say that models are able to perform as well as humans on the benchmarks tests, but subjectively we can see that they're not performing at human level at tasks in general. There is essentially an arms race between the emergent properties of LLMs, and the benchmarks that aim to measure them. Here are a couple of recent benchmarks that are pushing LLMs further. Massive Multitask Language Understanding, or MMLU, is designed specifically for modern LLMs. To perform well models must possess extensive world knowledge and problem-solving ability. Models are tested on elementary mathematics, US history, computer science, law, and more. In other words, tasks that extend way beyond basic language understanding. BIG-bench currently consists of 204 tasks, ranging through linguistics, childhood development, math, common sense reasoning, biology, physics, social bias, software development and more. BIG-bench comes in three different sizes, and part of the reason for this is to keep costs achievable, as running these large benchmarks can incur large inference costs. A final benchmark you should know about is the Holistic Evaluation of Language Models, or HELM. The HELM framework aims to improve the transparency of models, and to offer guidance on which models perform well for specific tasks. HELM takes a multimetric approach, measuring seven metrics across 16 core scenarios, ensuring that trade-offs between models and metrics are clearly exposed. One important feature of HELM is that it assesses on metrics beyond basic accuracy measures, like precision of the F1 score. The benchmark also includes metrics for fairness, bias, and toxicity, which are becoming increasingly important to assess as LLMs become more capable of human-like language generation, and in turn of exhibiting potentially harmful behavior. HELM is a living benchmark that aims to continuously evolve with the addition of new scenarios, metrics, and models. You can take a look at the results page to browse the LLMs that have been evaluated, and review scores that are pertinent to your project's needs.







- Full fine-tuning requires a significant amount of memory to allocate space for model weights, optimizer states, gradients, forward activations, and temporary memory throughout the training process.
- In contrast to full fine-tuning where every model weight is updated during supervised learning, parameter efficient fine tuning methods only update a small subset of parameters.
- Some PEFT techniques freeze most of the model weights and focus on fine tuning a subset of existing model parameters. for example, particular layers or components.
- Other techniques don't touch the original model weights at all, and instead add a small number of new parameters or layers and fine-tune only the new components. With PEFT, most if not all of the LLM weights are kept frozen. As a result, the number of trained parameters is much smaller than the number of parameters in the original LLM. In some cases, just 15-20% of the original LLM weights. This makes the memory requirements for training much more manageable.
- because the original LLM is only slightly modified or left unchanged, PEFT is less prone to the catastrophic forgetting problems of full fine-tuning.Full fine-tuning results in a new version of the model for every task you train on. Each of these is the same size as the original model, so it can create an expensive storage problem if you're fine-tuning for multiple tasks.
- With parameter efficient fine-tuning, you train only a small number of weights, which results in a much smaller footprint overall, as small as megabytes depending on the task. The new parameters are combined with the original LLM weights for inference. The PEFT weights are trained for each task and can be easily swapped out for inference, allowing efficient adaptation of the original model to multiple tasks.

![image](https://github.com/user-attachments/assets/28da8292-2572-4153-9714-67aaa83afa38)

- There are several methods you can use for parameter efficient fine-tuning, each with trade-offs on parameter efficiency, memory efficiency, training speed, model quality, and inference costs.
- Selective methods are those that fine-tune only a subset of the original LLM parameters. There are several approaches that you can take to identify which parameters you want to update. You have the option to train only certain components of the model or specific layers, or even individual parameter types. Researchers have found that the performance of these methods is mixed and there are significant trade-offs between parameter efficiency and compute efficiency.
- Reparameterization methods also work with the original LLM parameters, but reduce the number of parameters to train by creating new low rank transformations of the original network weights. A commonly used technique of this type is LoRA,
- Lastly, additive methods carry out fine-tuning by keeping all of the original LLM weights frozen and introducing new trainable components.
- Additive has two main approaches  Adapter methods add new trainable layers to the architecture of the model, typically inside the encoder or decoder components after the attention or feed-forward layers.  Soft prompt methods, on the other hand, keep the model architecture fixed and frozen, and focus on manipulating the input to achieve better performance. This can be done by adding trainable parameters to the prompt embeddings or keeping the input fixed and retraining the embedding weights.a specific soft prompts technique called prompt tuning. 

The input prompt is turned into tokens, which are then converted to embedding vectors and passed into the encoder and/or decoder parts of the transformer. In both of these components, there are two kinds of neural networks; self-attention and feedforward networks. The weights of these networks are learned during pre-training. After the embedding vectors are created, they're fed into the self-attention layers where a series of weights are applied to calculate the attention scores. During full fine-tuning, every parameter in these layers is updated.
LoRA is a strategy that reduces the number of parameters to be trained during fine-tuning by freezing all of the original model parameters and then injecting a pair of rank decomposition matrices alongside the original weights. The dimensions of the smaller matrices are set so that their product is a matrix with the same dimensions as the weights they're modifying. You then keep the original weights of the LLM frozen and train the smaller matrices using the same supervised learning process
For inference, the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights. You then add this to the original weights and replace them in the model with these updated values. You now have a LoRA fine-tuned model that can carry out your specific task.
Because this model has the same number of parameters as the original, there is little to no impact on inference latency. applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve performance gains. However, in principle, you can also use LoRA on other components like the feed-forward layers. But since most of the parameters of LLMs are in the attention layers, you get the biggest savings in trainable parameters by applying LoRA to these weights matrices. 

![image](https://github.com/user-attachments/assets/4a4eaaba-8b6f-4a9a-88de-4ec6b17d348f)

Since the rank-decomposition matrices are small, you can fine-tune a different set for each task and then switch them out at inference time by updating the weights.
Suppose you train a pair of LoRA matrices for a specific task; let's call it Task A. To carry out inference on this task, you would multiply these matrices together and then add the resulting matrix to the original frozen weights. You then take this new summed weights matrix and replace the original weights where they appear in your model. You can then use this model to carry out inference on Task A. If instead, you want to carry out a different task, say Task B, you simply take the LoRA matrices you trained for this task, calculate their product, and then add this matrix to the original weights and update the model again. The memory required to store these LoRA matrices is very small.
you can use LoRA to train for many tasks. Switch out the weights when you need to use them, and avoid having to store multiple full-size versions of the LLM.
how to choose the rank of the LoRA matrices. This is a good question and still an active area of research.
In principle, the smaller the rank, the smaller the number of trainable parameters, and the bigger the savings on compute. However, there are some issues related to model performance to consider The takeaway here is that ranks in the range of 4-32 can provide you with a good trade-off between reducing trainable parameters and preserving performance. 








## prompt Engineering:
Prompt tuning sounds a bit like prompt engineering, but they are quite different from each other. With prompt engineering, you work on the language of your prompt to get the completion you want. This could be as simple as trying different words or phrases or more complex, like including examples for one or Few-shot Inference. The goal is to help the model understand the nature of the task you're asking it to carry out and to generate a better completion. However, there are some limitations to prompt engineering, as it can require a lot of manual effort to write and try different prompts. You're also limited by the length of the context window, and at the end of the day, you may still not achieve the performance you need for your task. 

## Prompt Tuning:
- With prompt tuning, you add additional trainable tokens to your prompt and leave it up to the supervised learning process to determine their optimal values. The set of trainable tokens is called a soft prompt, and it gets prepended to embedding vectors that represent your input text. The soft prompt vectors have the same length as the embedding vectors of the language tokens. And including somewhere between 20 and 100 virtual tokens can be sufficient for good performance
- The tokens that represent natural language are hard in the sense that they each correspond to a fixed location in the embedding vector space. However, the soft prompts are not fixed discrete words of natural language. Instead, you can think of them as virtual tokens that can take on any value within the continuous multidimensional embedding space.
- And through supervised learning, the model learns the values for these virtual tokens that maximize performance for a given task. In full fine tuning, the training data set consists of input prompts and output completions or labels. The weights of the large language model are updated during supervised learning. In contrast with prompt tuning, the weights of the large language model are frozen and the underlying model does not get updated.
- Instead, the embedding vectors of the soft prompt gets updated over time to optimize the model's completion of the prompt. Prompt tuning is a very parameter efficient strategy because only a few parameters are being trained. In contrast with the millions to billions of parameters in full fine tuning, similar to what you saw with LoRA. You can train a different set of soft prompts for each task and then easily swap them out at inference time. You can train a set of soft prompts for one task and a different set for another. To use them for inference, you prepend your input prompt with the learned tokens to switch to another task, you simply change the soft prompt. Soft prompts are very small on disk, so this kind of fine tuning is extremely efficient and flexible.
- prompt tuning doesn't perform as well as full fine tuning for smaller LLMs. However, as the model size increases, so does the performance of prompt tuning. And once models have around 10 billion parameters, prompt tuning can be as effective as full fine tuning and offers a significant boost in performance over prompt engineering alone. One potential issue to consider is the interpretability of learned virtual tokens. because the soft prompt tokens can take any value within the continuous embedding vector space. The trained tokens don't correspond to any known token, word, or phrase in the vocabulary of the LLM. However, an analysis of the nearest neighbor tokens to the soft prompt location shows that they form tight semantic clusters.In other words, the words closest to the soft prompt tokens have similar meanings.
- LoRA is broadly used in practice because of the comparable performance to full fine tuning for many tasks

### RLHF (Reinforcement learning with human feedback):
- It helps to align the model with human values. LLMs might have a challenge in that it's creating sometimes harmful content or like a toxic tone or voice. And by aligning the model with human feedback and using reinforcement learning as an algorithm. You can help to align the model to reduce that and to align towards, less harmful content and much more helpful content as well.
- LLMs do generate problematic outputs. But it feels like with the progress of technology, researchers are consistently making them morethree Hs, (Honest, hopeful and harmless)

















### How to use LLMs as a reasoning engine and let it cause our own of routines to create an agent that can take actions:



### Responsible AI:

### RAG (Retrieval augmented system)
