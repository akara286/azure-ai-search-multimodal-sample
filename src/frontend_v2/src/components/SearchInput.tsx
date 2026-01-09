import React, { useState } from "react";
import { Button, Spinner, Tooltip } from "@fluentui/react-components";
import { Sparkle24Regular, Stop24Regular, Send24Filled, Mic24Regular } from "@fluentui/react-icons";
import "./SearchInput.css";

interface SearchInputProps {
    isLoading: boolean;
    onSearch: (query: string) => void;
    onStop?: () => void;
    currentStep?: string | null;
    modeName?: string;
}

const SearchInput: React.FC<SearchInputProps> = ({ isLoading, onSearch, onStop, currentStep, modeName }) => {
    const [query, setQuery] = useState("");
    const [isFocused, setIsFocused] = useState(false);

    const handleSearch = () => {
        if (query.trim()) {
            onSearch(query.trim());
            setQuery("");
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSearch();
        }
    };

    return (
        <div className={`search-wrapper ${isFocused ? "focused" : ""}`}>
            {isLoading && (
                <div className="status-indicator fade-in">
                    <Spinner size="tiny" />
                    <span className="status-text">{currentStep || "Thinking..."}</span>
                </div>
            )}

            <div className="input-bar">
                <div className="input-prefix">
                    <Tooltip content={modeName || "AI Search"} relationship="description">
                        <Sparkle24Regular className="ai-icon" />
                    </Tooltip>
                </div>

                <input
                    className="main-input"
                    type="text"
                    placeholder={`Ask anything...`}
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onFocus={() => setIsFocused(true)}
                    onBlur={() => setIsFocused(false)}
                    disabled={isLoading}
                />

                <div className="input-actions">
                    {isLoading ? (
                        <Button appearance="subtle" icon={<Stop24Regular />} onClick={onStop} className="action-btn stop-btn" title="Stop generating" />
                    ) : (
                        <>
                            {/* Placeholder for Voice Input if needed */}
                            {!query && <Button appearance="subtle" icon={<Mic24Regular />} className="action-btn" />}
                            <Button
                                appearance={query ? "primary" : "subtle"}
                                icon={<Send24Filled />}
                                onClick={handleSearch}
                                disabled={!query.trim()}
                                className={`action-btn send-btn ${query ? "active" : ""}`}
                            />
                        </>
                    )}
                </div>
            </div>

            <div className="disclaimer-text">AI-generated content may be incorrect • {modeName}</div>
        </div>
    );
};

export default SearchInput;
