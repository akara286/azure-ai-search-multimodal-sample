import React, { useEffect, useState } from "react";

import { Button, Drawer, DrawerBody, DrawerHeader, DrawerHeaderTitle, Spinner } from "@fluentui/react-components";
import { Dismiss20Regular, DocumentPdf20Regular, ArrowDownload20Regular } from "@fluentui/react-icons";

import { getCitationDocument } from "../api/api";
import { Citation } from "../api/models";
import "./CitationViewer.css";
import PdfHighlighter from "./PdfHighlighter";

interface Props {
    show: boolean;
    citation: Citation;
    toggle: () => void;
}

const CitationViewer: React.FC<Props> = ({ show, toggle, citation }) => {
    const [pdfPath, setPDFPath] = useState<string>("");
    const [fullPdfUrl, setFullPdfUrl] = useState<string>("");
    const [isLoading, setIsLoading] = useState<boolean>(false);

    useEffect(() => {
        if (show && citation) {
            setIsLoading(true);
            setPDFPath("");
            setFullPdfUrl("");

            // Fetch both page-specific and full PDF URLs
            const pageNumber = citation.locationMetadata?.pageNumber || 1;

            Promise.all([
                getCitationDocument(citation.title, pageNumber),
                getCitationDocument(citation.title, 1) // Get URL without page fragment for full PDF
            ])
                .then(([pageUrl, fullUrl]) => {
                    setPDFPath(pageUrl);
                    // Remove page fragment from full PDF URL if present
                    const baseUrl = fullUrl.split('#')[0];
                    setFullPdfUrl(baseUrl);
                })
                .finally(() => {
                    setIsLoading(false);
                });
        }
    }, [citation, show]);

    const handleViewFullPdf = () => {
        if (fullPdfUrl) {
            // Open PDF at the specific page using the #page=N fragment
            const pageNumber = citation.locationMetadata?.pageNumber || 1;
            window.open(`${fullPdfUrl}#page=${pageNumber}`, '_blank', 'noopener,noreferrer');
        }
    };

    const handleDownloadPdf = () => {
        if (fullPdfUrl) {
            const link = document.createElement('a');
            link.href = fullPdfUrl;
            link.download = citation.title || 'document.pdf';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    const pageNumber = citation.locationMetadata?.pageNumber || 1;

    return (
        <Drawer size="medium" position="end" separator open={show} onOpenChange={toggle} style={{ maxWidth: "550px" }}>
            <DrawerHeader>
                <DrawerHeaderTitle
                    action={
                        <div className="header-actions">
                            {fullPdfUrl && (
                                <>
                                    <Button
                                        appearance="subtle"
                                        aria-label="View Full PDF"
                                        icon={<DocumentPdf20Regular />}
                                        onClick={handleViewFullPdf}
                                        title="View Full PDF"
                                    />
                                    <Button
                                        appearance="subtle"
                                        aria-label="Download PDF"
                                        icon={<ArrowDownload20Regular />}
                                        onClick={handleDownloadPdf}
                                        title="Download PDF"
                                    />
                                </>
                            )}
                            <Button appearance="subtle" aria-label="Close" icon={<Dismiss20Regular />} onClick={toggle} />
                        </div>
                    }
                >
                    Citation
                </DrawerHeaderTitle>
            </DrawerHeader>

            <DrawerBody className="citation-viewer-body">
                <div className="citation-document-info">
                    <h3 className="citation-title">{citation.title}</h3>
                </div>

                {isLoading && (
                    <div className="citation-loading">
                        <Spinner size="medium" label="Loading document..." />
                    </div>
                )}

                {!isLoading && pdfPath && (
                    <div className="citation-pdf-container">
                        <PdfHighlighter
                            pdfPath={pdfPath}
                            pageNumber={pageNumber}
                        />
                    </div>
                )}
            </DrawerBody>
        </Drawer>
    );
};

export default CitationViewer;
