"use client";

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const sampleCounts = {
  current: { normal: 8202, anomaly: 32 },
  vibration: { normal: 1774, anomaly: 16 },
} as const;

type Sensor = keyof typeof sampleCounts;
type Dataset = keyof (typeof sampleCounts)[Sensor];

type Signal = {
  sensor: Sensor;
  dataset: Dataset;
  index: number;
  timestamp: string;
  frequencies: number[];
  values: number[];
};

export default function Dashboard() {
  const [sensor, setSensor] = useState<Sensor>("vibration");
  const [dataset, setDataset] = useState<Dataset>("anomaly");
  const [sampleIndex, setSampleIndex] = useState(0);
  const [signal, setSignal] = useState<Signal | null>(null);
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const chartData = useMemo(
    () =>
      signal?.frequencies.map((frequency, index) => ({
        frequency,
        value: signal.values[index],
      })) ?? [],
    [signal],
  );

  const maxIndex = sampleCounts[sensor][dataset] - 1;

  async function loadSignal() {
    setLoading(true);
    setError("");
    setReport("");

    try {
      const response = await fetch(
        `${API_URL}/signals/${sensor}/${dataset}/${sampleIndex}`,
      );
      if (!response.ok) throw new Error("데이터를 불러오지 못했습니다.");
      setSignal(await response.json());
    } catch (caught) {
      setSignal(null);
      setError(caught instanceof Error ? caught.message : "오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  }

  async function analyzeSignal() {
    if (!signal) return;
    setAnalyzing(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: `${signal.sensor} ${signal.dataset} 데이터의 ${signal.index}번 샘플(${signal.timestamp})을 분석하고 점검 리포트를 작성해줘.`,
        }),
      });
      if (!response.ok) throw new Error("에이전트 분석에 실패했습니다.");
      const result = (await response.json()) as { response?: string };
      setReport(result.response ?? "에이전트 응답이 없습니다.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "오류가 발생했습니다.");
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-slate-100 sm:px-8">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8">
          <p className="mb-2 text-sm font-semibold tracking-[0.18em] text-cyan-400">
            ROBOT WELDING MONITOR
          </p>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            설비 이상 진단 에이전트
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
            센서 샘플을 선택해 주파수 스펙트럼을 확인하고 LangGraph
            에이전트에게 분석을 요청하세요.
          </p>
        </header>

        <section className="mb-6 grid gap-4 rounded-2xl border border-slate-800 bg-slate-900 p-5 md:grid-cols-[1fr_1fr_1fr_auto] md:items-end">
          <label className="grid gap-2 text-sm text-slate-300">
            센서
            <select
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100"
              value={sensor}
              onChange={(event) => {
                setSensor(event.target.value as Sensor);
                setSampleIndex(0);
              }}
            >
              <option value="vibration">진동</option>
              <option value="current">전류</option>
            </select>
          </label>

          <label className="grid gap-2 text-sm text-slate-300">
            데이터셋
            <select
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100"
              value={dataset}
              onChange={(event) => {
                setDataset(event.target.value as Dataset);
                setSampleIndex(0);
              }}
            >
              <option value="normal">정상</option>
              <option value="anomaly">이상</option>
            </select>
          </label>

          <label className="grid gap-2 text-sm text-slate-300">
            샘플 번호 (0-{maxIndex})
            <input
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100"
              type="number"
              min={0}
              max={maxIndex}
              value={sampleIndex}
              onChange={(event) =>
                setSampleIndex(
                  Math.min(maxIndex, Math.max(0, Number(event.target.value))),
                )
              }
            />
          </label>

          <button
            className="rounded-lg bg-cyan-400 px-5 py-2.5 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={loading}
            onClick={loadSignal}
          >
            {loading ? "불러오는 중" : "데이터 불러오기"}
          </button>
        </section>

        {error && (
          <p className="mb-6 rounded-xl border border-red-900 bg-red-950/60 px-4 py-3 text-sm text-red-200">
            {error} FastAPI 서버가 실행 중인지 확인하세요.
          </p>
        )}

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(300px,0.8fr)]">
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
            <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-sm text-slate-400">주파수 스펙트럼</p>
                <h2 className="mt-1 text-xl font-semibold">
                  {signal
                    ? `${signal.sensor === "vibration" ? "진동" : "전류"} · ${signal.timestamp}`
                    : "샘플을 불러와 주세요"}
                </h2>
              </div>
              {signal && (
                <span
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    signal.dataset === "anomaly"
                      ? "bg-red-400/15 text-red-300"
                      : "bg-emerald-400/15 text-emerald-300"
                  }`}
                >
                  {signal.dataset === "anomaly" ? "이상 데이터" : "정상 데이터"}
                </span>
              )}
            </div>

            <div className="h-[390px] rounded-xl bg-slate-950/70 p-3">
              {signal ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="frequency"
                      stroke="#94a3b8"
                      tickLine={false}
                      tickCount={6}
                      type="number"
                      domain={["dataMin", "dataMax"]}
                      unit="Hz"
                    />
                    <YAxis stroke="#94a3b8" tickLine={false} width={72} />
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #334155",
                        borderRadius: 10,
                      }}
                      labelFormatter={(value) => `${Number(value).toFixed(2)} Hz`}
                    />
                    <Line
                      dataKey="value"
                      name={sensor === "vibration" ? "진동 (g)" : "전류 (A)"}
                      type="linear"
                      stroke="#22d3ee"
                      strokeWidth={1.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">
                  위에서 센서와 샘플을 선택하세요.
                </div>
              )}
            </div>
          </section>

          <section className="flex min-h-[500px] flex-col rounded-2xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-sm text-slate-400">에이전트 리포트</p>
            <h2 className="mt-1 text-xl font-semibold">이상 진단 및 점검 권고</h2>

            <div className="my-5 flex-1 rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-sm leading-7 text-slate-300">
              {report ? (
                <p className="whitespace-pre-wrap">{report}</p>
              ) : (
                <div className="space-y-4 text-slate-500">
                  <p>분석 결과가 이곳에 표시됩니다.</p>
                  <ul className="list-inside list-disc space-y-1">
                    <li>정상·이상 판정</li>
                    <li>관측된 신호 특징</li>
                    <li>가능한 원인 후보</li>
                    <li>권장 점검 사항</li>
                  </ul>
                </div>
              )}
            </div>

            <button
              className="w-full rounded-lg bg-white px-5 py-3 font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={!signal || analyzing}
              onClick={analyzeSignal}
            >
              {analyzing ? "에이전트 분석 중" : "에이전트에게 분석 요청"}
            </button>
          </section>
        </div>
      </div>
    </main>
  );
}
