import { lazy, Suspense } from "react";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ToastProvider } from "./components/Toast";
import { Spinner } from "./components/ui";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Register = lazy(() => import("./pages/Register"));
const Identify = lazy(() => import("./pages/Identify"));
const Compare = lazy(() => import("./pages/Compare"));
const Library = lazy(() => import("./pages/Library"));
const ObjectDetail = lazy(() => import("./pages/ObjectDetail"));
const HistoryPage = lazy(() => import("./pages/History"));
const ResearchLab = lazy(() => import("./pages/ResearchLab"));

const page = (el: JSX.Element) => <Suspense fallback={<Spinner label="Loading page…" />}>{el}</Suspense>;

export default function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={page(<Dashboard />)} />
            <Route path="register" element={page(<Register />)} />
            <Route path="identify" element={page(<Identify />)} />
            <Route path="compare" element={page(<Compare />)} />
            <Route path="library" element={page(<Library />)} />
            <Route path="library/:id" element={page(<ObjectDetail />)} />
            <Route path="history" element={page(<HistoryPage />)} />
            <Route path="research" element={page(<ResearchLab />)} />
            <Route path="*" element={<div className="py-20 text-center text-slate-400">Page not found. <Link className="text-accent" to="/">Go to overview</Link></div>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  );
}
