"""
Query Planner for intelligent query decomposition and optimization.

Uses GPT-5-mini to analyze complex queries and break them into
focused subqueries for better retrieval results.
"""

import json
import logging
from typing import List, Optional
from openai import AsyncAzureOpenAI

from models import Message, QueryPlan, SubQuery
from prompts import QUERY_PLANNER_PROMPT, SEARCH_QUERY_OPTIMIZATION_PROMPT

logger = logging.getLogger("backend_v2.query_planner")


class QueryPlanner:
    """
    Intelligent query planner using GPT-5-mini.

    Capabilities:
    - Analyzes query complexity
    - Decomposes complex queries into focused subqueries
    - Optimizes queries for search index retrieval
    - Considers chat history for context
    """

    def __init__(
        self,
        openai_client: AsyncAzureOpenAI,
        model_name: str = "gpt-5-mini",
    ):
        self.openai_client = openai_client
        self.model_name = model_name

    async def plan_query(
        self,
        user_message: str,
        chat_history: List[Message],
        max_subqueries: int = 4,
    ) -> QueryPlan:
        """
        Analyze the user query and create an execution plan.

        For simple queries, returns a single optimized subquery.
        For complex queries, decomposes into multiple focused subqueries.

        Args:
            user_message: The user's question
            chat_history: Previous conversation for context
            max_subqueries: Maximum number of subqueries to generate

        Returns:
            QueryPlan with decomposed subqueries and reasoning
        """
        logger.info(f"Planning query: {user_message[:100]}...")

        # Build the planning prompt
        history_context = self._format_chat_history(chat_history)

        planning_messages = [
            {"role": "system", "content": QUERY_PLANNER_PROMPT},
            {
                "role": "user",
                "content": f"""Chat history:
{history_context}

Current user question: {user_message}

Analyze this question and create a query plan with up to {max_subqueries} subqueries.
Return your response as a valid JSON object."""
            }
        ]

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=planning_messages,
                response_format={"type": "json_object"},
                max_completion_tokens=1000,
            )

            content = response.choices[0].message.content

            # Handle empty or None response
            if not content or not content.strip():
                logger.warning("Empty response from query planner, using original query")
                return QueryPlan(
                    original_query=user_message,
                    subqueries=[
                        SubQuery(
                            query=user_message,
                            intent="Direct search (empty planner response)",
                            filters=None,
                        )
                    ],
                    reasoning="Fallback due to empty planner response",
                )

            plan_data = json.loads(content)

            # Parse into QueryPlan model - filter out empty queries
            subqueries = [
                SubQuery(
                    query=sq.get("query", "").strip(),
                    intent=sq.get("intent", ""),
                    filters=sq.get("filters"),
                )
                for sq in plan_data.get("subqueries", [])
                if sq.get("query", "").strip()  # Only include non-empty queries
            ]

            # If no subqueries were generated, create one from the original
            if not subqueries:
                subqueries = [
                    SubQuery(
                        query=user_message,
                        intent="Direct search for user question",
                        filters=None,
                    )
                ]

            return QueryPlan(
                original_query=user_message,
                subqueries=subqueries[:max_subqueries],
                reasoning=plan_data.get("reasoning", "Single query execution"),
            )

        except Exception as e:
            logger.error(f"Query planning failed: {e}")
            # Fallback to simple single query
            return QueryPlan(
                original_query=user_message,
                subqueries=[
                    SubQuery(
                        query=user_message,
                        intent="Fallback direct search",
                        filters=None,
                    )
                ],
                reasoning=f"Fallback due to planning error: {str(e)}",
            )

    async def optimize_query(
        self,
        query: str,
        chat_history: List[Message],
    ) -> dict:
        """
        Optimize a single query for better search results.

        Adds synonyms, related terms, and contextual disambiguation.

        Args:
            query: The query to optimize
            chat_history: Chat history for context

        Returns:
            Dict with optimized_query and key_terms
        """
        history_context = self._format_chat_history(chat_history)

        optimization_messages = [
            {"role": "system", "content": SEARCH_QUERY_OPTIMIZATION_PROMPT},
            {
                "role": "user",
                "content": f"""Chat history:
{history_context}

Query to optimize: {query}

Return a JSON object with the optimized query."""
            }
        ]

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=optimization_messages,
                response_format={"type": "json_object"},
                max_completion_tokens=200,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            return {
                "optimized_query": result.get("optimized_query", query),
                "key_terms": result.get("key_terms", []),
            }

        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return {
                "optimized_query": query,
                "key_terms": [],
            }

    async def should_decompose(
        self,
        user_message: str,
        chat_history: List[Message],
    ) -> bool:
        """
        Determine if a query is complex enough to warrant decomposition.

        Simple heuristics + LLM judgment for edge cases.

        Args:
            user_message: The user's question
            chat_history: Chat history for context

        Returns:
            True if query should be decomposed
        """
        # Simple heuristics first
        complexity_indicators = [
            " and " in user_message.lower(),
            " or " in user_message.lower(),
            " compare " in user_message.lower(),
            " difference " in user_message.lower(),
            " vs " in user_message.lower(),
            user_message.count("?") > 1,
            len(user_message.split()) > 20,
        ]

        # If multiple indicators, definitely decompose
        if sum(complexity_indicators) >= 2:
            return True

        # If no indicators, probably simple
        if sum(complexity_indicators) == 0:
            return False

        # Edge case: ask the LLM
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You analyze queries for a search system. Respond with only 'yes' or 'no'."
                    },
                    {
                        "role": "user",
                        "content": f"Is this query complex enough to benefit from being split into multiple search queries? Query: {user_message}"
                    }
                ],
                max_completion_tokens=10,
            )

            answer = response.choices[0].message.content.strip().lower()
            return answer == "yes"

        except Exception as e:
            logger.warning(f"Complexity check failed: {e}")
            return False

    def _format_chat_history(self, chat_history: List[Message]) -> str:
        """Format chat history for inclusion in prompts."""
        if not chat_history:
            return "No previous conversation."

        formatted = []
        for msg in chat_history[-6:]:  # Last 6 messages for context
            role = msg.get("role", "user")
            content_parts = msg.get("content", [])
            text = " ".join(
                part.get("text", "") for part in content_parts
                if isinstance(part, dict)
            )
            formatted.append(f"{role.capitalize()}: {text}")

        return "\n".join(formatted)
