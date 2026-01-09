import { useEffect, useState } from "react";
import { getConfig } from "../api/api";
import { OpenAIAPIMode, SearchConfig } from "../api/models";

export default function useConfig() {
    const [config, setConfig] = useState<SearchConfig>({
        use_semantic_ranker: false,
        chunk_count: 10,
        openai_api_mode: OpenAIAPIMode.ChatCompletions,
        use_streaming: true,
        use_knowledge_agent: true
    });

    const [indexes, setIndexes] = useState<string[]>([]);
    const [knowledgeBaseAvailable, setKnowledgeBaseAvailable] = useState<boolean>(false);
    const [configError, setConfigError] = useState<string | null>(null);
    const [isConfigLoading, setIsConfigLoading] = useState<boolean>(true);

    useEffect(() => {
        const fetchConfig = async () => {
            setIsConfigLoading(true);
            setConfigError(null);
            try {
                const configData = await getConfig();
                setIndexes(configData.indexes);
                setKnowledgeBaseAvailable(configData.knowledgeBaseAvailable);
            } catch (error) {
                console.error("Failed to fetch config:", error);
                setConfigError("Unable to connect to the server. Please check your connection and refresh the page.");
            } finally {
                setIsConfigLoading(false);
            }
        };

        fetchConfig();
    }, []);

    return { config, setConfig, indexes, knowledgeBaseAvailable, configError, isConfigLoading };
}
