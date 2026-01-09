import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Badge, Button, Tooltip } from "@fluentui/react-components";
import { Copy24Regular, BrainCircuit24Regular, Person24Regular, Bot24Regular } from "@fluentui/react-icons";

import { ProcessingStepsMessage, RoleType, Thread, ThreadType, Citation } from "../api/models";
import "./ChatContent.css";
import Citations from "./Citations";
import ProcessingSteps from "./ProcessingSteps";
import CitationViewer from "./CitationViewer";

interface Props {
    processingStepMsg: Record<string, ProcessingStepsMessage[]>;
    thread: Thread[];
}

const ChatContent: React.FC<Props> = ({ thread, processingStepMsg }) => {
    const [showProcessingSteps, setShowProcessingSteps] = useState(false);
    const [processRequestId, setProcessRequestId] = useState("");
    const [selectedCitation, setSelectedCitation] = useState<Citation | undefined>();
    const [isCitationViewerOpen, setIsCitationViewerOpen] = useState(false);
    const [highlightedCitation, setHighlightedCitation] = useState<string | undefined>();
    const [showCopied, setShowCopied] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const messageToBeCopied: Record<string, string> = {};

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [thread]);

    const messagesGroupedByRequestId = Object.values(
        thread.reduce((acc: { [key: string]: Thread[] }, message: Thread) => {
            if (!acc[message.request_id]) {
                acc[message.request_id] = [];
            }
            acc[message.request_id].push(message);
            return acc;
        }, {})
    );

    // Citations helper
    const citationRegex = /\[([^\]]+)\]/g;

    // Find citation by index or content_id
    const findCitation = (label: string, textCitations?: Citation[]) => {
        if (!textCitations || textCitations.length === 0) return undefined;

        // First try to find by numerical index (for [1], [2] style citations)
        const index = parseInt(label) - 1;
        if (!isNaN(index) && index >= 0 && index < textCitations.length) {
            return textCitations[index];
        }

        // Otherwise search by content_id (for hash-style citations)
        return textCitations.find(c => c.content_id === label || c.docId === label);
    };

    // Get citation number from content_id
    const getCitationNumber = (contentId: string, textCitations?: Citation[]) => {
        if (!textCitations) return null;
        const index = textCitations.findIndex(c => c.content_id === contentId || c.docId === contentId);
        return index >= 0 ? index + 1 : null;
    };

    const citationHit = (label: string, docId: string, message: Thread) => {
        // Convert hash to numbered reference
        const citationNum = getCitationNumber(docId, message.textCitations) || label;

        return (
            <sup
                key={label}
                onMouseLeave={() => setHighlightedCitation(undefined)}
                onMouseEnter={() => setHighlightedCitation(docId)}
                onClick={() => {
                    const citation = findCitation(docId, message.textCitations);
                    if (citation) {
                        setSelectedCitation(citation);
                        setIsCitationViewerOpen(true);
                    }
                }}
                className="citation-sup"
                title="View Citation"
            >
                [{citationNum}]
            </sup>
        );
    };

    const renderWithCitations = (children: React.ReactNode, message: Thread) => {
        return React.Children.map(children, child => {
            if (typeof child === "string") {
                return child.split(citationRegex).map((part, index) => {
                    if (index % 2 === 0) return part;
                    return citationHit(part, part, message);
                });
            }
            return child;
        });
    };

    // Adjusted wrapper to pass message context
    const MarkdownRenderer = ({ content, message }: { content: string; message: Thread }) => (
        <ReactMarkdown
            components={{
                p: ({ children }) => <p>{renderWithCitations(children, message)}</p>,
                li: ({ children }) => <li>{renderWithCitations(children, message)}</li>,
                h1: ({ children }) => <h3>{renderWithCitations(children, message)}</h3>,
                h2: ({ children }) => <h4>{renderWithCitations(children, message)}</h4>,
                h3: ({ children }) => <h5>{renderWithCitations(children, message)}</h5>,
                strong: ({ children }) => <strong>{renderWithCitations(children, message)}</strong>,
                em: ({ children }) => <em>{renderWithCitations(children, message)}</em>,
                blockquote: ({ children }) => <blockquote>{renderWithCitations(children, message)}</blockquote>,
                span: ({ children }) => <span>{renderWithCitations(children, message)}</span>
            }}
            remarkPlugins={[remarkGfm]}
        >
            {content}
        </ReactMarkdown>
    );

    const getCurProcessingStep = (requestId: string): Record<string, ProcessingStepsMessage[]> => {
        return { [requestId]: processingStepMsg[requestId] };
    };

    return (
        <div className="chat-container">
            {messagesGroupedByRequestId.map((group, groupIndex) => (
                <div key={groupIndex} className="message-group fade-in-up">
                    {group.map((message, msgIndex) => {
                        const isUser = message.role === RoleType.User;

                        // Store answer for copy functionality
                        if (message.type === ThreadType.Answer) {
                            messageToBeCopied[message.request_id] = message.answerPartial?.answer || "";
                        }

                        if (message.type === ThreadType.Info) return null; // Skip info messages for cleaner UI

                        return (
                            <div key={msgIndex} className={`message-row ${isUser ? "user-row" : "bot-row"}`}>
                                <div className="message-avatar">{isUser ? <Person24Regular /> : <Bot24Regular />}</div>

                                <div className={`message-bubble ${isUser ? "user-bubble" : "bot-bubble"}`}>
                                    {message.type === ThreadType.Message && <div className="message-text">{message.message}</div>}

                                    {message.type === ThreadType.Answer && (
                                        <div className="markdown-content">
                                            {message.retrievalModeLabel && (
                                                <Badge
                                                    appearance="tint"
                                                    color={message.retrievalMode === "knowledge_base" ? "brand" : "important"}
                                                    className="mode-badge"
                                                >
                                                    {message.retrievalModeLabel}
                                                </Badge>
                                            )}
                                            <MarkdownRenderer content={message.answerPartial?.answer || ""} message={message} />
                                        </div>
                                    )}

                                    {message.type === ThreadType.Error && <div className="error-content">{message.message || "An error occurred."}</div>}

                                    {/* Action Bar for Bot Messages */}
                                    {!isUser && (message.type === ThreadType.Answer || message.type === ThreadType.Error) && (
                                        <div className="message-actions">
                                            <Tooltip content={showCopied ? "Copied!" : "Copy"} relationship="label">
                                                <Button
                                                    appearance="subtle"
                                                    size="small"
                                                    icon={<Copy24Regular />}
                                                    onClick={() => {
                                                        const text = messageToBeCopied[message.request_id] || "";
                                                        navigator.clipboard.writeText(text);
                                                        setShowCopied(true);
                                                        setTimeout(() => setShowCopied(false), 2000);
                                                    }}
                                                />
                                            </Tooltip>

                                            <Tooltip content="Show Thought Process" relationship="label">
                                                <Button
                                                    appearance="subtle"
                                                    size="small"
                                                    icon={<BrainCircuit24Regular />}
                                                    disabled={!processingStepMsg[message.request_id]}
                                                    onClick={() => {
                                                        setProcessRequestId(message.request_id);
                                                        setShowProcessingSteps(true);
                                                    }}
                                                />
                                            </Tooltip>
                                        </div>
                                    )}

                                    {/* Citations Section */}
                                    {!isUser && message.type === ThreadType.Answer && (
                                        <div className="citations-area">
                                            <Citations
                                                imageCitations={message.imageCitations || []}
                                                textCitations={message.textCitations || []}
                                                highlightedCitation={highlightedCitation}
                                                onCitationClick={citation => {
                                                    setSelectedCitation(citation);
                                                    setIsCitationViewerOpen(true);
                                                }}
                                            />
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            ))}
            <div ref={messagesEndRef} />

            <ProcessingSteps
                showProcessingSteps={showProcessingSteps}
                processingStepMsg={getCurProcessingStep(processRequestId)}
                toggleEditor={() => setShowProcessingSteps(!showProcessingSteps)}
            />
            {selectedCitation && <CitationViewer show={isCitationViewerOpen} toggle={() => setIsCitationViewerOpen(false)} citation={selectedCitation} />}
        </div>
    );
};

export default ChatContent;
