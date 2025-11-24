#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ROOT
from ROOT import RooRealVar,RooBreitWigner,RooNumConvPdf,RooGaussian,RooExponential,RooDataSet
from ROOT import RooDataHist,RooAbsData,RooPlot,RooChebychev,RooAddPdf,RooArgList,RooPolynomial,TH1F,RooFit,RooArgSet, RooRandom
import os, json, platform, subprocess, time, re, getpass
import pandas as pd
import matplotlib.pyplot as plt
import datetime, psutil, sys
from datetime import datetime
import matplotlib.lines as mlines

# Suppress ROOT/RooFit output
ROOT.gErrorIgnoreLevel = ROOT.kFatal
ROOT.Math.MinimizerOptions.SetDefaultPrintLevel(-1)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)


system = platform.system()  # "Linux", "Darwin", "Windows", ...

# Helper to get a human-friendly OS name and version
def get_os_name_version():
    """
    Return a human-friendly (os_name, os_version), e.g.
    - macOS + numeric version on Darwin
    - Distro name + version on Linux (from /etc/os-release)
    - Windows + release on Windows
    """
    sysname = platform.system()
    # macOS
    if sysname == "Darwin":
        # mac_ver()[0] gives the macOS version (e.g. '15.0.1')
        release, _, _ = platform.mac_ver()
        os_name = "macOS"
        os_version = release or platform.release()
        return os_name, os_version

    # Linux
    if sysname == "Linux":
        os_name = "Linux"
        os_version = platform.release()
        try:
            with open("/etc/os-release", "r") as f:
                data = f.read().splitlines()
            name_val = None
            version_val = None
            for line in data:
                line = line.strip()
                if line.startswith("NAME=") and name_val is None:
                    name_val = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("VERSION_ID=") and version_val is None:
                    version_val = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("VERSION=") and version_val is None:
                    version_val = line.split("=", 1)[1].strip().strip('"')
            if name_val:
                os_name = name_val
            if version_val:
                os_version = version_val
        except Exception:
            # fall back to defaults if /etc/os-release is not readable
            pass
        return os_name, os_version

    # Windows
    if sysname == "Windows":
        try:
            out = subprocess.check_output(
                "wmic os get Caption,Version /value",
                shell=True
            ).decode(errors="ignore")

            os_name = None
            os_version = None

            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Caption="):
                    os_name = line.split("=", 1)[1].strip()
                elif line.startswith("Version="):
                    os_version = line.split("=", 1)[1].strip()

            if os_name and os_version:
                # Example:
                #   os_name    = "Microsoft Windows 11 Pro"
                #   os_version = "10.0.26100"
                return os_name, os_version
        except Exception:
            pass

        os_name = "Windows"
        os_version = platform.version()  
        return os_name, os_version


    # Fallback for other systems
    return sysname, platform.release()


# In[ ]:


def get_cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            model = subprocess.check_output("lscpu | grep 'Model name:'", shell=True).decode()
            return model.split(":", 1)[1].strip()
        elif system == "Darwin":  # macOS
            model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode()
            return model.strip()
        elif system == "Windows":
            model = subprocess.check_output("wmic cpu get name", shell=True).decode().split("\n")[1]
            return model.strip()
        else:
            return "Unknown CPU"
    except Exception as e:
        return f"Error: {e}"

def get_cpu_info():
    """
    Return a dict with basic CPU info, branching by OS.
    Keys used later in the script:
      - physical_cores
      - total_core_count
      - logical_cpus (optional, may be None)
    """
    info = {
        "physical_cores": None,
        "total_core_count": None,
        "logical_cpus": None,
        "threads_per_core": None,
        "perf_cores": None,
        "eff_cores": None,
        "ht_enabled": None,
    }

    if system == "Linux":
        # Use lscpu output and sched_getaffinity as before
        def parse_lscpu():
            out = subprocess.check_output("lscpu", shell=True).decode()
            lines = out.split("\n")
            parsed = {}
            for line in lines:
                if "Core(s) per socket" in line:
                    parsed["cores_per_socket"] = int(line.split()[-1])
                elif "Socket(s):" in line:
                    parsed["sockets"] = int(line.split()[-1])
                elif "Thread(s) per core" in line:
                    parsed["threads_per_core"] = int(line.split()[-1])
                elif line.startswith("CPU(s):"):
                    try:
                        parsed["logical_cpus"] = int(line.split()[1])
                    except ValueError:
                        # If CPU range is printed (e.g. '0-95,192-287') skip
                        continue
            return parsed

        cpu_info = parse_lscpu()
        # Available cores constrained by affinity mask
        try:
            available_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            # Fallback if sched_getaffinity is not available
            available_cores = os.cpu_count() or 1
        cores_per_socket = cpu_info.get("cores_per_socket")
        sockets         = cpu_info.get("sockets")
        logical_cpus    = cpu_info.get("logical_cpus") or available_cores
        threads_per_core = cpu_info.get("threads_per_core")

        if cores_per_socket and sockets:
            physical_cores = cores_per_socket * sockets
        else:
            physical_cores = logical_cpus

        # CORRECTION FOR HYBRID / ARM-LIKE SITUATIONS:
        # - No Hyperthreading (threads_per_core == 1)
        # - “physical” is illogically smaller than “logical”

        if threads_per_core == 1 and physical_cores < logical_cpus:
            physical_cores = logical_cpus

        info["physical_cores"] = physical_cores
        info["total_core_count"] = available_cores
        info["logical_cpus"] = logical_cpus
        info["threads_per_core"] = threads_per_core
        if threads_per_core is not None:
            info["ht_enabled"] = threads_per_core > 1
        else:
            info["ht_enabled"] = None

    elif system == "Darwin":
        # macOS: use sysctl to get physical and logical cores
        try:
            phys = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
        except Exception:
            phys = os.cpu_count() or 1
        try:
            logical = int(subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"]).decode().strip())
        except Exception:
            logical = os.cpu_count() or phys

        # Try to detect performance / efficiency cores if available
        perf_cores = None
        eff_cores = None
        try:
            # Scan sysctl -a output for keys like:
            #   hw.perflevel0.physicalcpu: 8
            # and ignore ...physicalcpu_max, etc.
            out = subprocess.check_output(["sysctl", "-a"]).decode()
            for line in out.splitlines():
                line = line.strip()
                # We only want lines that start with "hw.perflevelX.physicalcpu:"
                m = re.match(r'^hw\.perflevel(\d+)\.physicalcpu:\s*(\d+)', line)
                if not m:
                    continue
                level = int(m.group(1))
                try:
                    val = int(m.group(2))
                except Exception:
                    continue
                if level == 0:
                    perf_cores = val if perf_cores is None else perf_cores + val
                else:
                    eff_cores = (eff_cores or 0) + val
        except Exception:
            # If anything goes wrong, leave perf_cores/eff_cores as None
            pass

        # Threads per core and hyperthreading
        threads_per_core = None
        if phys:
            threads_per_core = logical / float(phys)

        info["physical_cores"] = phys
        info["total_core_count"] = logical
        info["logical_cpus"] = logical
        info["perf_cores"] = perf_cores
        info["eff_cores"] = eff_cores
        info["threads_per_core"] = threads_per_core
        if threads_per_core is not None:
            info["ht_enabled"] = threads_per_core > 1.0
        else:
            info["ht_enabled"] = None

    elif system == "Windows":
        # Windows: try to get physical cores and logical processors via wmic
        logical = os.cpu_count() or 1
        physical = None
        try:
            out = subprocess.check_output(
                "wmic cpu get NumberOfCores,NumberOfLogicalProcessors /value",
                shell=True
            ).decode()
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("NumberOfCores"):
                    try:
                        physical = int(line.split("=", 1)[1])
                    except Exception:
                        continue
        except Exception:
            pass
        if physical is None:
            physical = logical

        info["physical_cores"] = physical
        info["total_core_count"] = logical
        info["logical_cpus"] = logical
        if physical:
            threads_per_core = logical / float(physical)
        else:
            threads_per_core = None
        info["threads_per_core"] = threads_per_core
        if threads_per_core is not None:
            info["ht_enabled"] = threads_per_core > 1.0
        else:
            info["ht_enabled"] = None

    else:
        # Unknown OS: minimal info
        logical = os.cpu_count() or 1
        info["physical_cores"] = logical
        info["total_core_count"] = logical
        info["logical_cpus"] = logical
        info["threads_per_core"] = None
        info["ht_enabled"] = None

    return info


model_name = get_cpu_model()
cpu_info = get_cpu_info()
core_count = cpu_info["total_core_count"]
physical_cores = cpu_info["physical_cores"]
total_core_count = cpu_info["total_core_count"]
logical_cpus = cpu_info["logical_cpus"]
threads_per_core = cpu_info["threads_per_core"]
perf_cores = cpu_info["perf_cores"]
eff_cores = cpu_info["eff_cores"]
ht_enabled = cpu_info["ht_enabled"]

# OS name and version, username
os_name, os_version = get_os_name_version()
username = getpass.getuser()
machine_name = platform.node()

# In[ ]:


def add_common_footer(df):
    """
    Add a horizontal separator line and three text blocks under the current
    matplotlib figure, using information from the first row of df.
    """
    row0 = df.iloc[0]

    col1 = (
        f"Available cores: {row0['Available_cores']}\n"
        f"Total physical cores: {row0['Total_physical_cores']}\n"
        f"Total threads: {row0['Total_threads']}"
    )

    col2 = (
        f"Event size: {row0['event_size']}\n"
        f"Seed number: {row0['seed']}\n"
        f"Number of bins: {row0['bins']}"
    )

    col3 = (
        f"ROOT version: {row0['ROOT_version']}\n"
        f"Python version: {row0['Python_version']}\n"
        f"Username: {row0['Username']}\n"
        f"Machine: {row0['Machine']}"
    )

    fig = plt.gcf()

    # Reserve a modest margin at the bottom (~18% of the figure height)
    plt.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))

    # Horizontal separator line just above the footer area
    line = mlines.Line2D(
        [0.05, 0.95],   # from 5% to 95% of figure width
        [0.19, 0.19],   # slightly below the x-axis tick labels
        transform=fig.transFigure,
        color='black',
        linewidth=1.0,
        alpha=0.7
    )
    fig.add_artist(line)

    # Three text blocks in a single row under the plot
    fig.text(0.05, 0.11, col1, ha='left', va='center', fontsize=9)
    fig.text(0.40, 0.11, col2, ha='left', va='center', fontsize=9)
    fig.text(0.75, 0.11, col3, ha='left', va='center', fontsize=9)


# Helper to format current time as 'YYYY-MM-DD HH:MM:SS' (no microseconds)
def fmt_now():
    """Return current time as 'YYYY-MM-DD HH:MM:SS' (no microseconds)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


bins = 200
seed = 1010
events = 100000

# Convert event count to a string label
if events == 100:
    name = "100"
elif events == 1000:
    name = "1k"
elif events == 10000:
    name = "10k"
elif events == 100000:
    name = "100k"
elif events == 500000:
    name = "500k"
elif events == 1000000:
    name = "1M"
elif events == 5000000:
    name = "5M"
elif events == 10000000:
    name = "10M"
elif events == 50000000:
    name = "50M"
elif events == 100000000:
    name = "100M"
elif events == 500000000:
    name = "500M"
elif events == 1000000000:
    name = "1000M"
else:
    name = str(events)

#
# Results directory structure:
# results/<event_label>/data_dir for CSV/JSON
# results/<event_label>/plot_dir for PNG and topology
base_results_dir = "results"
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_model = model_name.replace(" ", "_").replace("/", "_")
# run_label encodes CPU model, username, event label, thread count and timestamp
run_label = f"{safe_model}_{username}_{name}_{total_core_count}threads_{run_timestamp}"

event_dir = os.path.join(base_results_dir, name)

data_dir = os.path.join(event_dir, "data_dir")
plot_dir = os.path.join(event_dir, "plot_dir")

json_dir = os.path.join(data_dir, "json")
csv_dir = os.path.join(data_dir, "csv")
logs_dir = os.path.join(data_dir, "logs")
txt_dir = os.path.join(data_dir, "txt")

for d in (event_dir, data_dir, plot_dir, json_dir, csv_dir, logs_dir, txt_dir):
    os.makedirs(d, exist_ok=True)

# Redirect stdout/stderr to a log file (tee: console + file)
log_path = os.path.join(logs_dir, f"log_{run_label}.log")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
        for s in self.streams:
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open(log_path, "a")
sys.stdout = Tee(sys.stdout, _log_file)
sys.stderr = Tee(sys.stderr, _log_file)

benchmark_start = datetime.now()
print("=== Benchmark run started ===", benchmark_start.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print(f"""--- CPU Information ---
OS: {os_name}
OS version: {os_version}
Model: {model_name}
Available Cores (for this process): {core_count}
Username: {username}
""")
print(f"Physical cores: {physical_cores}")
if logical_cpus is not None:
    if logical_cpus == total_core_count:
        print(f"Total threads (logical cores): {total_core_count}")
    else:
        print("Total thread count and logical CPU count differ")
        print(f"Total threads (os.cpu_count / affinity): {total_core_count}")
        print(f"Logical cores (from system query): {logical_cpus}")
if threads_per_core is not None:
    print(f"Threads per core: {threads_per_core}")
if ht_enabled is not None:
    print(f"Hyperthreading enabled: {ht_enabled}")
if perf_cores is not None or eff_cores is not None:
    print(f"Performance cores: {perf_cores}")
    print(f"Efficiency cores: {eff_cores}")
print("", flush=True)

# Try to dump CPU topology if lstopo is available (Linux/macOS/etc.)
try:
    import shutil
    if shutil.which("lstopo") is not None:
        topology_svg = os.path.join(plot_dir, f"cpu_topology_{run_label}.svg")
        print(f"Exporting CPU topology with lstopo to {topology_svg}", flush=True)
        try:
            subprocess.run(["lstopo", topology_svg], check=True)
            print("CPU topology export completed successfully.", flush=True)
        except Exception as e:
            print(f"Could not run lstopo (skipping topology export): {e}", flush=True)
    else:
        print("lstopo not found in PATH, skipping CPU topology export.", flush=True)
except Exception as e:
    print(f"Error while checking for lstopo: {e}", flush=True)



# Save a JSON config with system and benchmark settings
run_config = {
    "OS_Name": os_name,
    "OS_Version": os_version,
    "Model": model_name,
    "Username": username,
    "Machine": machine_name,
    "Available_cores_for_process": core_count,
    "Physical_cores": physical_cores,
    "Logical_cpus": logical_cpus,
    "Threads_per_core": threads_per_core,
    "Performance_cores": perf_cores,
    "Efficiency_cores": eff_cores,
    "Hyperthreading_enabled": ht_enabled,
    "Events": events,
    "Bins": bins,
    "Seed": seed,
    "ROOT_version": ROOT.gROOT.GetVersion(),
    "Python_version": sys.version.split()[0],
}
config_path = os.path.join(json_dir, f"run_config_{run_label}.json")
with open(config_path, "w") as cfg_f:
    json.dump(run_config, cfg_f, indent=2)


# In[ ]:


# Observable definition
xvar = RooRealVar("xvar", "", -10, 10)
xvar.setBins(bins)

# Breit Wigner Signal //
mean = RooRealVar("m", "mean", 0.2, -1, 1)                      # Breit Wigner mean
gamma = RooRealVar("#Gamma", "gamma", 2, 0.1, 5)               # Breit Wigner width
signal = RooBreitWigner("BW", "BW signal", xvar, mean, gamma)  # Breit Wigner PDF

# Gaussian Resolution Function //
zero = RooRealVar("zero", "Gaussian resolution mean", 0.)                     # Offset from mean
sigma = RooRealVar("#sigma", "sigma", 1.5, 0.1, 5)                            # Gaussian sigma
resol = RooGaussian("resol", "Gaussian resolution", xvar, zero, sigma)       # Gaussian PDF

# Background //
alpha = RooRealVar("#alpha", "Exponential Parameter", -0.05, -2.0, 0.0)
bkg = RooExponential("Bkg", "Bkg", xvar, alpha)

# Gaussian + BW convolution //
convolution = RooNumConvPdf("convolution", "BW (X) gauss", xvar, signal, resol)

# TotalPdf = Gaussian + Bkg //
sigfrac = RooRealVar("sig1frac", "fraction of component 1 in signal", 0.5, 0., 1.)
total = RooAddPdf("totalPDF", "totalPDF", RooArgList(convolution, bkg), sigfrac)

print(f"\n------Generating {name} events\n")

# Generating data
RooRandom.randomGenerator().SetSeed(seed)
print("Generating dataset...", fmt_now(), flush=True)
data = total.generate(xvar, events)
print("Dataset generation done.", fmt_now(), flush=True)

# Save generated data to file
#data_file = f"{data_dir}{events}_events.txt"
#data.write(data_file)

print(f"\nFitting {name} events\n")


# In[ ]:


#cpu_counts = [1,2]     
cpu_counts = list(range(1, core_count + 1))

results = []

results_csv = os.path.join(csv_dir, f"fit_timing_results_{run_label}.csv")
if os.path.exists(results_csv):
    os.remove(results_csv)


print('Run started at:', fmt_now(), flush=True)

proc = psutil.Process()

ref_wall, ref_cpu = None, None

for ncpu in cpu_counts:
    print(f"[NumCPU={ncpu}] Fit started at {fmt_now()}", flush=True)

    nll = total.createNLL(data, ROOT.RooFit.NumCPU(ncpu))
    minimizer= ROOT.RooMinimizer(nll)
    minimizer.setPrintLevel(-1)
    minimizer.setVerbose(False)
        
    now = datetime.now()
    
    _start_date = now.strftime("%Y_%b_%d")   # 2025_Feb_03
    _start__time  = now.strftime("%H:%M:%S")   # 14:05:33

    wall_start = time.perf_counter()
    cpu_start = proc.cpu_times()

    minimizer.migrad()
    minimizer.hesse()

    wall_end = time.perf_counter()
    cpu_end = proc.cpu_times()

    now_2 = datetime.now()
    
    _finish_date = now_2.strftime("%Y_%b_%d")   
    _finish__time  = now_2.strftime("%H:%M:%S")   

    wall_time = wall_end - wall_start
    user_cpu = cpu_end.user - cpu_start.user
    sys_cpu = cpu_end.system - cpu_start.system
    total_cpu = user_cpu + sys_cpu

    if ref_wall is None:
        ref_wall = wall_time
        # Baseline CPU: first sys, or total, or user
        if sys_cpu > 0:
            ref_cpu = sys_cpu
        elif total_cpu > 0:
            ref_cpu = total_cpu
        else:
            ref_cpu = user_cpu

    #speedup_wall = ref_wall / wall_time
    #speedup_cpu = ref_cpu / sys_cpu
    # Wall time speedup 
    speedup_wall = ref_wall / wall_time if wall_time > 0 else float("nan")

    # CPU time speedup: sys_cpu might be 0 in some platforms
    if sys_cpu > 0 and ref_cpu > 0:
        speedup_cpu = ref_cpu / sys_cpu
    else:
        speedup_cpu = float("nan")    
    results.append({
        "NumCPU": ncpu,
        "User_CPU(s)": user_cpu,
        "System_CPU(s)": sys_cpu,
        "Total_CPU(s)": total_cpu,
        "Wall_Time(s)": wall_time,
        "Speedup_Wall": speedup_wall,
        "Speedup_CPU": speedup_cpu,
        "Model": model_name,
        "OS_Name": os_name,
        "OS_Version": os_version,
        "Username": username,
        "Available_cores": core_count,
        "Total_physical_cores": physical_cores,
        "Total_threads": logical_cpus if logical_cpus is not None else total_core_count,
        "Machine": machine_name,
        "event_size": name,
        "seed": seed,
        "bins": bins,
        "Start_Date": _start_date, 
        "Start_Time": _start__time,         
        "Finish_Date": _finish_date, 
        "Finish_Time": _finish__time, 
        "ROOT_version": ROOT.gROOT.GetVersion(),
        "Python_version": sys.version.split()[0],
        "Threads_per_core": threads_per_core,
        "Performance_cores": perf_cores,
        "Efficiency_cores": eff_cores,
        "Hyperthreading_enabled": ht_enabled,
    })
    df_row = pd.DataFrame([results[-1]])
    file_exists = os.path.isfile(results_csv)
    df_row.to_csv(
        results_csv,
        mode="a",              
        header=not file_exists, 
        index=False
    )

    print(
        f"[NumCPU={ncpu}] Finished at {fmt_now()} | "
        f"wall={wall_time:.2f}s user={user_cpu:.2f}s sys={sys_cpu:.2f}s",
        flush=True)
    print("", flush=True)
    del speedup_wall, speedup_cpu, total_cpu,sys_cpu,user_cpu,wall_time,cpu_end,wall_end,minimizer,cpu_start,wall_start,nll

df = pd.DataFrame(results)




# In[ ]:


plt.figure(figsize=(15, 7))
plt.plot(df["NumCPU"], df["Wall_Time(s)"], marker='.', markersize=14, color='tab:green', alpha=0.4, label="Wall time")
plt.xlabel("Number of CPUs")
plt.ylabel("Time (seconds)")
plt.title(f"Wall-clock Time vs Number of CPUs for {name} events in {model_name}")
plt.xticks(df["NumCPU"].unique(),rotation=60)
plt.legend()
plt.grid(True, alpha=0.3)
add_common_footer(df)
plt.savefig(os.path.join(plot_dir, f"wall_time_vs_numcpu_{run_label}.png"), bbox_inches="tight")
plt.close()



plt.figure(figsize=(15, 7))
plt.plot(df["NumCPU"], df["System_CPU(s)"], marker='*', markersize=14, color='tab:red', alpha=0.4, label="Sys time")
plt.xlabel("Number of CPUs")
plt.ylabel("Time (seconds)")
plt.title(f"System Time vs Number of CPUs for {name} events in {model_name}")
plt.xticks(df["NumCPU"].unique(),rotation=60)
plt.legend()
plt.grid(True, alpha=0.3)
add_common_footer(df)
plt.savefig(os.path.join(plot_dir, f"sys_time_vs_numcpu_{run_label}.png"), bbox_inches="tight")
plt.close()


plt.figure(figsize=(15, 7))
plt.plot(df["NumCPU"], df["Speedup_Wall"], marker='o', markersize=14, color='tab:blue', alpha=0.4, label="Speedup_Wall")
plt.xlabel("Number of CPUs")
plt.ylabel("Time (seconds)")
plt.title(f"Speedup Wall Time vs Number of CPUs for {name} events in {model_name}")
plt.xticks(df["NumCPU"].unique(),rotation=60)
plt.legend()
plt.grid(True, alpha=0.3)
add_common_footer(df)
plt.savefig(os.path.join(plot_dir, f"speedup_wall_vs_numcpu_{run_label}.png"), bbox_inches="tight")
plt.close()



baseline_row = df.loc[df["NumCPU"].idxmin()]
baseline = float(baseline_row["Wall_Time(s)"])
row_maxN = df.loc[df["NumCPU"].idxmax()]
N_max = int(row_maxN["NumCPU"])
S_max_obs = float(row_maxN["Speedup_Wall"])  
# Amdahl parameters
f_parallel = 1.0 - 1.0 / S_max_obs
serial_fraction = 1.0 - f_parallel
print(f"Observed max speedup S(N_max={N_max}) = {S_max_obs:.3f}")
print(f"Estimated parallel fraction f ~= {f_parallel:.3f}")
print(f"Estimated serial fraction 1-f ~= {serial_fraction:.3f}")

N_values = df["NumCPU"].astype(float).values
f = f_parallel
T_amdahl = baseline * ((1.0 - f) + f / N_values) #Expected walltime according to Amdahl model
df["Amdahl_Wall_Time"] = T_amdahl
df["Amdahl_Speedup"] = baseline / T_amdahl #Expected speedup according to Amdahl model

df.to_csv(results_csv, index=False)



plt.figure(figsize=(15, 7))
plt.plot(df["NumCPU"], df["Wall_Time(s)"], marker="o", linestyle="-", label="Measured wall time")
plt.plot(df["NumCPU"], df["Amdahl_Wall_Time"], marker="s", linestyle="--", label="Amdahl model")
plt.xlabel("Number of CPUs")
plt.ylabel("Time (seconds)")
plt.title(f"Wall time vs Amdahl model for {df['event_size'].iloc[0]} events")
plt.xticks(df["NumCPU"].unique(), rotation=60)
plt.grid(True, alpha=0.3)
plt.legend()
add_common_footer(df)
plt.savefig(os.path.join(plot_dir, f"amdahl_wall_time_{run_label}.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(15, 7))
plt.plot(df["NumCPU"], df["Speedup_Wall"], marker="o", linestyle="-", label="Measured speedup")
plt.plot(df["NumCPU"], df["Amdahl_Speedup"], marker="s", linestyle="--", label="Amdahl model")
plt.xlabel("Number of CPUs")
plt.ylabel("Speedup S(N)")
plt.title(f"Speedup vs Amdahl model for {df['event_size'].iloc[0]} events")
plt.xticks(df["NumCPU"].unique(), rotation=60)
plt.grid(True, alpha=0.3)
plt.legend()
add_common_footer(df)
plt.savefig(os.path.join(plot_dir, f"amdahl_speedup_{run_label}.png"), bbox_inches="tight")
plt.close()

# Print total benchmark duration in the log
end_time = datetime.now()
try:
    elapsed = end_time - benchmark_start
    elapsed_str = str(elapsed).split(".")[0]  # drop microseconds
    print(f"=== Benchmark completed successfully in {elapsed_str} ===", flush=True)
except Exception:
    pass

if "_log_file" in globals():
    try:
        _log_file.close()
    except Exception:
        pass
