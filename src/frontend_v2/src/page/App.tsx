import { FluentProvider, webDarkTheme, webLightTheme, Spinner, MessageBar, MessageBarBody, Title1, Text } from "@fluentui/react-components";
import { BrainCircuit24Regular } from "@fluentui/react-icons";
import { Layout } from "../components/Layout";
import ChatContent from "../components/ChatContent";
import { NavBar } from "../components/NavBar";
import Samples from "../components/Samples";
import SearchInput from "../components/SearchInput";
import useChat from "../hooks/useChat";
import useConfig from "../hooks/useConfig";
import useTheme from "../hooks/useTheme";
import { IntroTitle, IntroText } from "../api/defaults";
import "./App.css";

function App() {
    const { config, configError, isConfigLoading } = useConfig();
    const { thread, processingStepsMessage, isLoading, currentStep, handleQuery, onNewChat, stopQuery } = useChat(config);
    const modeName = "Knowledge Base";
    const { darkMode, setDarkMode } = useTheme();

    // Sidebar content
    const sidebar = <NavBar onNewChat={onNewChat} />;

    return (
        <FluentProvider theme={darkMode ? webDarkTheme : webLightTheme}>
            <Layout sidebar={sidebar} darkMode={darkMode} toggleDarkMode={() => setDarkMode(!darkMode)}>
                <div className="content-container">
                    {thread.length > 0 ? (
                        <div className="chat-interface">
                            <div className="chat-scroll-area">
                                <ChatContent thread={thread} processingStepMsg={processingStepsMessage} />
                            </div>
                            <div className="input-area">
                                <SearchInput onSearch={handleQuery} isLoading={isLoading} onStop={stopQuery} currentStep={currentStep} modeName={modeName} />
                            </div>
                        </div>
                    ) : (
                        <div className="welcome-screen">
                            <div className="hero-content">
                                <div className="hero-icon"><BrainCircuit24Regular /></div>
                                <Title1 align="center" className="hero-title">
                                    {IntroTitle}
                                </Title1>
                                <Text align="center" className="hero-subtitle">
                                    {IntroText}
                                </Text>

                                {isConfigLoading && <Spinner label="Initializing..." size="large" />}
                                {configError && (
                                    <MessageBar intent="error">
                                        <MessageBarBody>{configError}</MessageBarBody>
                                    </MessageBar>
                                )}

                                <div className="hero-samples">
                                    <Samples handleQuery={handleQuery} />
                                </div>

                                <div className="hero-input">
                                    <SearchInput
                                        onSearch={handleQuery}
                                        isLoading={isLoading}
                                        onStop={stopQuery}
                                        currentStep={currentStep}
                                        modeName={modeName}
                                    />
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </Layout>
        </FluentProvider>
    );
}

export default App;
