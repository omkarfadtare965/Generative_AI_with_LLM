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
- 





















