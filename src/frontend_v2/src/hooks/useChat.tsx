import { useState, useEffect, useRef } from "react";
import { sendChatApi } from "../api/api";
import { Thread, ProcessingStepsMessage, Chat, ThreadType, RoleType, SearchConfig } from "../api/models";

// Custom hook for managing chat state
export default function useChat(config: SearchConfig) {
    const [chatId, setChatId] = useState<string>();
    const [thread, setThread] = useState<Thread[]>([]);
    const [processingStepsMessage, setProcessingStepsMessage] = useState<Record<string, ProcessingStepsMessage[]>>({});
    const [chats, setChats] = useState<Record<string, Chat>>();
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [currentStep, setCurrentStep] = useState<string | null>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    const refreshChats = async () => {
        setChats({});
    };

    const stopQuery = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
            setIsLoading(false);
            setCurrentStep(null);
        }
    };

    const handleQuery = async (query: string) => {
        // Abort any existing request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        // Create new abort controller for this request
        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;

        setIsLoading(true);
        setCurrentStep("Starting...");
        try {
            const request_id = new Date().getTime().toString();

            if (!chatId) setChatId(request_id);

            const chatThread = thread
                .filter(message => message.role === "user" || message.role === "assistant")
                .map(msg => ({
                    role: msg.role,
                    content: [
                        {
                            text: msg.role === "assistant" ? msg.answerPartial?.answer : msg.message,
                            type: "text"
                        }
                    ]
                }));

            setThread(prevThread => {
                const newThread = [...prevThread, { request_id, type: ThreadType.Message, message: query, role: RoleType.User }];
                return newThread;
            });

            refreshChats();

            await sendChatApi(
                query,
                request_id,
                chatThread,
                config,
                message => {
                    if (message.event === "processing_step") {
                        const newStep = JSON.parse(message.data);
                        // Update current step for loading indicator
                        if (newStep.processingStep?.title) {
                            setCurrentStep(newStep.processingStep.title);
                        }
                        setProcessingStepsMessage(steps => {
                            const updatedSteps = { ...steps };
                            updatedSteps[newStep.request_id] = [...(steps[newStep.request_id] || []), newStep];
                            return updatedSteps;
                        });
                    } else if (message.event === "[END]") {
                        setIsLoading(false);
                        setCurrentStep(null);
                    } else {
                        const data = JSON.parse(message.data);
                        data.type = message.event;

                        setThread(prevThread => {
                            const index = prevThread.findIndex(msg => msg.message_id === data.message_id);
                            if (index !== -1) {
                                // Update existing message, merging new data over old
                                const newThread = [...prevThread];
                                newThread[index] = { ...prevThread[index], ...data };
                                newThread.sort((a, b) => new Date(a.request_id).getTime() - new Date(b.request_id).getTime());
                                return newThread;
                            } else {
                                // Add new message
                                const newThread = [...prevThread, data];
                                newThread.sort((a, b) => new Date(a.request_id).getTime() - new Date(b.request_id).getTime());
                                return newThread;
                            }
                        });
                    }
                },
                err => {
                    // Ignore abort errors
                    if (err instanceof DOMException && err.name === "AbortError") {
                        return;
                    }
                    console.error(err);
                    throw err;
                },
                signal
            );
        } catch (err) {
            // Ignore abort errors
            if (err instanceof DOMException && err.name === "AbortError") {
                return;
            }
            console.error(err);
            // Add error message to thread
            const request_id = new Date().getTime().toString();
            setThread(prevThread => [
                ...prevThread,
                {
                    request_id,
                    type: ThreadType.Error,
                    message: "Something went wrong. Please try again.",
                    role: RoleType.Assistant,
                    message_id: `error-${request_id}`
                }
            ]);
        } finally {
            setIsLoading(false);
            setCurrentStep(null);
            abortControllerRef.current = null;
        }
    };

    const onNewChat = () => {
        stopQuery(); // Stop any ongoing query when starting new chat
        setChatId(undefined);
        setThread([]);
    };

    useEffect(() => {
        refreshChats();
    }, [config]);

    return { chatId, thread, processingStepsMessage, chats, isLoading, currentStep, handleQuery, onNewChat, stopQuery };
}
