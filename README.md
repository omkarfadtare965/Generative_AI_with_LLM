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


## Techniques to reduce the memory required for training:







