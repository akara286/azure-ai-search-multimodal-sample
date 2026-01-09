import { ReactNode, useState } from "react";
import { Button, Tooltip } from "@fluentui/react-components";
import { Navigation24Regular, WeatherMoon24Regular, WeatherSunny24Regular } from "@fluentui/react-icons";
import "./Layout.css";

interface LayoutProps {
    children: ReactNode;
    sidebar: ReactNode;
    darkMode: boolean;
    toggleDarkMode: () => void;
}

export const Layout = ({ children, sidebar, darkMode, toggleDarkMode }: LayoutProps) => {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

    return (
        <div className={`app-layout ${darkMode ? "dark" : "light"}`} data-theme={darkMode ? "dark" : "light"}>
            <header className="app-header glass-panel">
                <div className="header-left">
                    <Button appearance="subtle" icon={<Navigation24Regular />} onClick={toggleSidebar} className="sidebar-toggle" />
                    <div className="brand">
                        <span className="brand-logo">✨</span>
                        <span className="brand-text">Exploration Archives</span>
                    </div>
                </div>

                <div className="header-right">
                    <Tooltip content={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"} relationship="description">
                        <Button appearance="subtle" icon={darkMode ? <WeatherSunny24Regular /> : <WeatherMoon24Regular />} onClick={toggleDarkMode} />
                    </Tooltip>
                </div>
            </header>

            <div className="app-body">
                <aside className={`app-sidebar glass-panel ${isSidebarOpen ? "open" : "closed"}`}>{sidebar}</aside>

                <main className="app-content">{children}</main>
            </div>
        </div>
    );
};
