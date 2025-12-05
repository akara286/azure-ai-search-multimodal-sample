import asyncio
import os
import logging
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.indexes.aio import SearchIndexClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def list_agents():
    # Load environment variables
    load_dotenv(dotenv_path="../../.azure/test/.env")

    endpoint = os.environ["SEARCH_SERVICE_ENDPOINT"]
    credential = DefaultAzureCredential()
    client = SearchIndexClient(endpoint=endpoint, credential=credential)

    try:
        print(f"Listing agents on {endpoint}...")
        agents = client.list_agents()
        found = False
        async for agent in agents:
            print(f"Found Agent: {agent.name}")
            if agent.name == os.environ["KNOWLEDGE_AGENT_NAME"]:
                found = True

        if not found:
            print(
                f"Agent '{os.environ['KNOWLEDGE_AGENT_NAME']}' NOT found in the list."
            )
        else:
            print(f"Agent '{os.environ['KNOWLEDGE_AGENT_NAME']}' FOUND.")

    except Exception as e:
        print(f"Error listing agents: {e}")
    finally:
        await client.close()
        await credential.close()


if __name__ == "__main__":
    asyncio.run(list_agents())
