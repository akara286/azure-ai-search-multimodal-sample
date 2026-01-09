export enum ThreadType {
    Answer = "answer",
    Message = "message",
    Citation = "citation",
    Info = "info",
    Error = "error"
}

export enum RoleType {
    User = "user",
    Assistant = "assistant"
}

export enum OpenAIAPIMode {
    ChatCompletions = "chat_completions"
}

export interface KnowledgeAgentMessage {
    role: RoleType;
    content: string;
}

export interface Thread {
    message?: string;
    request_id: string;
    message_id?: string;
    type: ThreadType;
    answerPartial?: { answer: string };
    log_json?: string;
    role: RoleType;
    textCitations?: Citation[];
    imageCitations?: Citation[];
    knowledgeAgentMessage?: KnowledgeAgentMessage;
    retrievalMode?: string;
    retrievalModeLabel?: string;
}

export type Coordinates = { x: number; y: number };

export type BoundingPolygon = Coordinates[];

export interface Citation {
    docId: string;
    content_id: string;
    title: string;
    text?: string;
    locationMetadata: {
        pageNumber: number;
        boundingPolygons: string;
    };
}

export interface Chat {
    name: string;
    thread: Thread[];
    id: string;
    lastUpdated: number;
}

export enum ProcessingStepType {
    Text = "text"
}

export interface ProcessingStep {
    title: string;
    description: string;
    type: ProcessingStepType;
    content: string;
}

export interface ProcessingStepsMessage {
    message_id: string;
    request_id: string;
    processingStep: ProcessingStep;
}

export interface SearchConfig {
    use_semantic_ranker: boolean;
    chunk_count: number;
    openai_api_mode: OpenAIAPIMode;
    use_streaming: boolean;
    use_knowledge_agent: boolean;
}
