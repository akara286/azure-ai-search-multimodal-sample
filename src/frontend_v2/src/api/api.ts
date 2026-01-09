import { EventSourceMessage, fetchEventSource } from "@microsoft/fetch-event-source";

import { SearchConfig } from "./models";

const sendChatApi = async (
    message: string,
    requestId: string,
    chatThread: any,
    config: SearchConfig,
    onMessage: (message: EventSourceMessage) => void,
    onError?: (err: unknown) => void,
    signal?: AbortSignal
) => {
    const endpoint = "/v2/chat";

    await fetchEventSource(endpoint, {
        openWhenHidden: true,
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: message, request_id: requestId, chatThread: chatThread, config }),
        onerror: (err) => {
            // Handle network errors more gracefully
            if (err instanceof Error && err.message.includes("network")) {
                console.error("Network error - connection may have timed out:", err);
            }
            if (onError) onError(err);
        },
        onmessage: (msg) => {
            // Ignore heartbeat comments (they start with ":")
            if (msg.event === "" && msg.data === "") {
                return;
            }
            onMessage(msg);
        },
        signal
    });
};

interface ConfigResponse {
    indexes: string[];
    knowledgeBaseAvailable: boolean;
    model: string;
    features: {
        knowledge_base_api: boolean;
        query_rewrite: boolean;
        semantic_ranking: boolean;
        hybrid_search: boolean;
    };
}

const getConfig = async (): Promise<ConfigResponse> => {
    const response = await fetch(`/v2/config`);
    const data = await response.json();
    return {
        indexes: [data.search_index],
        knowledgeBaseAvailable: data.features?.knowledge_base_api ?? false,
        model: data.model,
        features: data.features ?? {}
    };
};

// Legacy function for backward compatibility
const listIndexes = async () => {
    const config = await getConfig();
    return { indexes: config.indexes };
};

const getCitationDocument = async (fileName: string, page: number = 1): Promise<string> => {
    try {
        const response = await fetch(`/v2/citation/${encodeURIComponent(fileName)}?page=${page}`);
        if (!response.ok) {
            console.error(`Failed to get citation document: ${response.status}`);
            return "";
        }
        const data = await response.json();
        return data.url || "";
    } catch (err) {
        console.error("Error fetching citation document:", err);
        return "";
    }
};

export { sendChatApi, listIndexes, getConfig, getCitationDocument };
export type { ConfigResponse };
