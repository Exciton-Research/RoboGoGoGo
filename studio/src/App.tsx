import { WorkspaceProvider } from "./state/WorkspaceProvider";
import { StudioHeader } from "./components/StudioHeader";
import { WorkspaceSidebar } from "./components/WorkspaceSidebar";
import { SimulationStage } from "./components/SimulationStage";
import { Inspector } from "./components/Inspector";
import { StatusBar } from "./components/StatusBar";
import { HelpDialog } from "./components/HelpDialog";

export default function App() {
  return (
    <WorkspaceProvider>
      <div className="app">
        <StudioHeader />
        <main className="workbench">
          <WorkspaceSidebar />
          <SimulationStage />
          <Inspector />
        </main>
        <StatusBar />
        <HelpDialog />
      </div>
    </WorkspaceProvider>
  );
}
