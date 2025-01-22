## api_key = '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'

import os

from groq import Groq

client = Groq(
    # This is the default and can be omitted
    api_key = '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',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Você é um assistente jurídico."
        },
        {
            "role": "user",
            "content": "Explique quais são os requisitos de um contrato consumeirista",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)