import asyncio
import os
import logging
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    KnowledgeAgent as AzureSearchKnowledgeAgent,
    KnowledgeAgentTargetIndex,
    KnowledgeAgentAzureOpenAIModel,
    AzureOpenAIVectorizerParameters,
)
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_agent():
    # Load environment variables
    load_dotenv(dotenv_path="../../.azure/test/.env")

    endpoint = os.environ["SEARCH_SERVICE_ENDPOINT"]
    index_name = os.environ["SEARCH_INDEX_NAME"]
    agent_name = os.environ["KNOWLEDGE_AGENT_NAME"]
    openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    # Assuming the model name for the agent is the same as the deployment or defined elsewhere
    # In app.py/knowledge_agent.py it uses chatcompletions_model_name which comes from AZURE_OPENAI_MODEL_NAME
    openai_model_name = os.environ["AZURE_OPENAI_MODEL_NAME"]

    print(f"Endpoint: {endpoint}")
    print(f"Index Name: {index_name}")
    print(f"Agent Name: {agent_name}")

    credential = DefaultAzureCredential()
    client = SearchIndexClient(endpoint=endpoint, credential=credential)

    try:
        print(f"Creating/Updating agent: {agent_name}...")
        agent = AzureSearchKnowledgeAgent(
            name=agent_name,
            target_indexes=[
                KnowledgeAgentTargetIndex(
                    index_name=index_name,
                    default_include_reference_source_data=True,
                )
            ],
            models=[
                KnowledgeAgentAzureOpenAIModel(
                    azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                        resource_url=openai_endpoint,
                        deployment_name=openai_deployment,
                        model_name=openai_model_name,
                    )
                )
            ],
        )
        
        await client.create_or_update_agent(agent)
        print("Agent created successfully!")
    except Exception as e:
        print(f"Error creating agent: {e}")
    finally:
        await client.close()
        await credential.close()

if __name__ == "__main__":
    asyncio.run(create_agent())
