import "./globals.css";
import { Sidebar } from "../components/shell";
export const metadata={title:"WattWise.ai",description:"AI Energy Digital Twin"};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body><div className="app"><Sidebar/>{children}</div></body></html>}
