# AI Recipe Search Agent

This project is an **AI-powered recipe search agent** that allows users to find recipes based on their preferences (ingredients, cooking time, calories, keywords, etc.).  
It uses **LlamaIndex**, **LangChain**, and **OpenAI GPT-3.5** to index and query recipes from a CSV file.

---

## Features
- Load recipes from a CSV file.
- Convert recipe data into vector-based searchable documents.
- Create and persist a recipe index for quick searches.
- Query recipes in natural language using an AI agent.
- Supports search by:
  - Recipe name
  - Ingredients
  - Category (e.g., dessert, vegan)
  - Cooking time
  - Calories

---

## Project Structure
- recipe_index/ # Persistent storage for the index
- recipes.csv # Your recipe dataset
- recipe_agent.py # Main Python script
- README.md # Project documentation
