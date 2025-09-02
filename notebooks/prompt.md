Please walk me through and explain the process of using a chroma_db vectorstore as context with qlora adapters to fine-tune the model output.

I am trying to use QLoRA adapters to fine tune output for the particular user's style of writing and use chroma_db for context (model awareness, previous conversation awareness, relationship awareness, other long-term memory)

I believe I need 500 to 1000 Q & A pairs for model fine-tuning to create the style of writing of the user. 
I want to give the model self awareness, so this would be like finding a biography online of an individual and adding that to the chroma_db as a description and adding personal stories of users.

To create more data for Q & A pairs for model fine-tuning, I would need to use reinforcement learning to label the original data with largeer weight than data that is generated from a larger language model. What is this process?

please create examples of using a chroma_db to add context to the model so the model seems to take on the personality of a particular person.

This is how I have used the vectorstore and the qlora adapter together. Please explain this clearly step-by-step. I need to know that is personal to the particular individual to give the impression of self-awareness and long-term memory (what is unique to the chroma_db vectorstore), and what is personal to the model fine-tuning process. For example, I know a base language model (such as meta-llama/Llama-3.2-1B-Instruct) is non-unique to the model fine-tuning process. However, the 500 - 1000 Q & A pairs of documents used to fine-tune the model and the model adapter (meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8) are unique to the model with respect to fine tuning. I need to save the base language model to the docker image and store the adapter and documents in s3 storage. What is the chroma_db vectorstore analogue to this?