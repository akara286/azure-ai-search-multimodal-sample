import React from "react";
import { Chat24Regular } from "@fluentui/react-icons";
import samplesData from "../content/samples.json";
import "./Samples.css";

interface Props {
    handleQuery: (q: string, isNew?: boolean) => void;
}

const Samples: React.FC<Props> = ({ handleQuery }) => {
    const samples: string[] = samplesData.queries;

    return (
        <div className="samples-grid">
            {samples?.map((sample, index) => (
                <div key={index} className="sample-card glass-panel" onClick={() => handleQuery(sample)} role="button" tabIndex={0}>
                    <div className="sample-icon">
                        <Chat24Regular />
                    </div>
                    <span className="sample-text">{sample}</span>
                </div>
            ))}
        </div>
    );
};

export default Samples;
