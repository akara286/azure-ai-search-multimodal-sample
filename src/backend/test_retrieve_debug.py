import asyncio
import os
import logging
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_retrieve():
    # Load environment variables
    load_dotenv(dotenv_path="../../.azure/test/.env")

    endpoint = os.environ["SEARCH_SERVICE_ENDPOINT"]
    agent_name = os.environ["KNOWLEDGE_AGENT_NAME"]
    index_name = os.environ["SEARCH_INDEX_NAME"]

    print(f"Testing retrieve on {endpoint} for agent {agent_name}...")

    credential = DefaultAzureCredential()
    client = KnowledgeAgentRetrievalClient(
        endpoint=endpoint, credential=credential, agent_name=agent_name
    )

    try:
        print("Sending retrieve request...")
        result = await client.retrieve(
            retrieval_request={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "how do I use segy snapshot tool?", "type": "text"}
                        ],
                    }
                ],
                "target_index_params": [
                    {
                        "indexName": index_name,
                        "includeReferenceSourceData": False,
                    }
                ],
            }
        )
        print("Retrieve successful!")
        if result.references:
            first_ref = result.references[0]
            print(f"First Reference: {first_ref}")
            # Try to access as_dict if available, or attributes
            try:
                print(f"First Reference Dict: {first_ref.as_dict()}")
            except:
                print("Could not convert to dict")
        else:
            print("No references found.")
    except Exception as e:
        print(f"Error calling retrieve: {e}")
    finally:
        await client.close()
        await credential.close()


if __name__ == "__main__":
    asyncio.run(test_retrieve())
