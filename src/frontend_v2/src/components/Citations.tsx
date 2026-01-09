import React from "react";

import { Caption1Strong, InteractionTag, InteractionTagPrimary } from "@fluentui/react-components";
import { ImageCopy24Filled, Text12Filled } from "@fluentui/react-icons";

import { Citation } from "../api/models";
import CitationViewer from "./CitationViewer";
import "./Citations.css";

interface CitationsProps {
    highlightedCitation?: string;
    imageCitations: Citation[];
    textCitations: Citation[];
    onCitationClick?: (citation: Citation) => void;
}

const Citations: React.FC<CitationsProps> = ({ imageCitations, textCitations, highlightedCitation, onCitationClick }) => {
    // Keep internal state for backward compatibility or independent usage if onCitationClick not provided
    const [citationsView, setCitationsView] = React.useState(false);
    const [selectedCitation, setSelectedCitation] = React.useState<Citation>();

    const handleCitationClick = (citation: Citation) => {
        if (onCitationClick) {
            onCitationClick(citation);
        } else {
            setSelectedCitation(citation);
            setCitationsView(true);
        }
    };

    const truncateText = (maxLength: number, text?: string) => {
        if (!text) return "";
        if (text.length <= maxLength) {
            return text;
        }
        return text.substring(0, maxLength) + "...";
    };

    return (
        <>
            <div>
                {!!(textCitations.length || imageCitations.length) && (
                    <>
                        <Caption1Strong className="citations-title" block>
                            Citations
                        </Caption1Strong>
                        <div className="citations">
                            {!!textCitations.length &&
                                textCitations.map((citation, index) => (
                                    <InteractionTag
                                        className="citation-interaction"
                                        key={index}
                                        appearance={
                                            highlightedCitation === `${citation.docId}` || highlightedCitation === String(index + 1) ? "brand" : "filled"
                                        }
                                    >
                                        <InteractionTagPrimary
                                            onClick={() => handleCitationClick(citation)}
                                            icon={<span className="citation-number">[{index + 1}]</span>}
                                        >
                                            {truncateText(40, citation.text || citation.title)}
                                        </InteractionTagPrimary>
                                    </InteractionTag>
                                ))}
                            {!!imageCitations.length &&
                                imageCitations.map((citation, index) => (
                                    <InteractionTag
                                        key={index}
                                        className="citation-interaction"
                                        appearance={highlightedCitation === `${citation.docId}` ? "brand" : "filled"}
                                    >
                                        <InteractionTagPrimary
                                            onClick={() => handleCitationClick(citation)}
                                            icon={<ImageCopy24Filled className="citation-image-icon" />}
                                        >
                                            {truncateText(40, citation.title)}
                                        </InteractionTagPrimary>
                                    </InteractionTag>
                                ))}
                        </div>
                    </>
                )}
            </div>
            {/* Only show internal viewer if we are managing state locally */}
            {!onCitationClick && selectedCitation && <CitationViewer show={citationsView} toggle={() => setCitationsView(false)} citation={selectedCitation} />}
        </>
    );
};

export default Citations;
