import { Document, Page } from "react-pdf";
import { pdfjs } from "react-pdf";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

interface PdfHighlighterProps {
    pdfPath: string;
    pageNumber: number;
}

const PdfHighlighter = ({ pdfPath, pageNumber }: PdfHighlighterProps) => {
    return (
        <div style={{ position: "relative" }}>
            <Document file={pdfPath}>
                <Page renderTextLayer={false} pageNumber={pageNumber} renderAnnotationLayer={false} />
            </Document>
        </div>
    );
};

export default PdfHighlighter;
