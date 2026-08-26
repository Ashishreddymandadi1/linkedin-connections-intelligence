import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createBrowserRouter, Navigate } from "react-router-dom";
import "./index.css";
import App from "./App";
import UploadPage from "./pages/UploadPage";
import EnrichmentPage from "./pages/EnrichmentPage";
import DashboardPage from "./pages/DashboardPage";
import SearchPage from "./pages/SearchPage";
import ResultsPage from "./pages/ResultsPage";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Navigate to="/upload" replace /> },
      { path: "upload", element: <UploadPage /> },
      { path: "datasets/:datasetId/enriching", element: <EnrichmentPage /> },
      { path: "datasets/:datasetId", element: <DashboardPage /> },
      { path: "datasets/:datasetId/search", element: <SearchPage /> },
      { path: "datasets/:datasetId/search/:searchId", element: <ResultsPage /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>,
);
