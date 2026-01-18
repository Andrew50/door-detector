import React from "react";
import ReactDOM from "react-dom/client";
import { withStreamlitConnection } from "streamlit-component-lib";

import { PdfJsViewer } from "./pdfjs_viewer";

const Connected = withStreamlitConnection(PdfJsViewer);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Connected />
  </React.StrictMode>
);

