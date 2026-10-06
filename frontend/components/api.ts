export const API=process.env.NEXT_PUBLIC_API_URL||"http://localhost:8000/api";

async function request<T>(path:string,init:RequestInit={}):Promise<T>{
	let lastError:unknown;
	for(let attempt=0;attempt<3;attempt++){
		const controller=new AbortController();
		const timer=setTimeout(()=>controller.abort(),8000);
		try{
			const r=await fetch(API+path,{...init,signal:controller.signal,cache:"no-store"});
			if(!r.ok)throw new Error((await r.json().catch(()=>null))?.detail||`Request failed (${r.status})`);
			return await r.json();
		}catch(error){
			lastError=error;
			if(attempt<2)await new Promise(resolve=>setTimeout(resolve,500*(attempt+1)));
		}finally{clearTimeout(timer)}
	}
	if(lastError instanceof DOMException&&lastError.name==='AbortError')throw new Error('Backend request timed out. Start FastAPI on port 8000.');
	throw lastError instanceof Error?lastError:new Error('Unable to connect to backend. Start FastAPI on port 8000.');
}

export function get<T>(path:string):Promise<T>{return request<T>(path)}
export function post<T>(path:string,data:unknown):Promise<T>{return request<T>(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)})}
