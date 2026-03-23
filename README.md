---
title: SHL Assessment Recommender
emoji: ":dart:"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SHL Assessment Recommendation System

AI-powered system that recommends relevant SHL assessments based on job descriptions or natural language queries.

## API Endpoints

- GET /health - Health check
- POST /recommend - Get assessment recommendations

## Usage

POST /recommend
{
    "query": "I need a Java developer who can collaborate with teams"
}
