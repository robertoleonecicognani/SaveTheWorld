s.boot;

(
b = Buffer.alloc(s,512,1,{|z|z.sine1Msg(1.0/[1,3,5,7,9,11,13,15,17])});
c = Buffer.read(s, "/path"); // Not needed a path, but still the variable c is necessary

fork{
	s.sync;
	~a = {arg amp = 1;
		var sig;
		var local;
		// TCHoose select randomly an output based on a (trigger, array of inputs)
		var f = [30,60,15]*Lag.kr(TChoose.kr(Impulse.kr(0.05),[0.75, 0.5, 1]), 8);

		sig = f;

		local = sig + LocalIn.ar(2);
		15.do({
			local = AllpassN.ar(local, 0.06, Rand(0.001, 0.06), 3)
		});
		LocalOut.ar(local*0.4);

		Out.ar(0, Limiter.ar(LPF.ar(local * amp, 4000), 0.8)*EnvGen.kr(Env([0, 1, 1, 0],[9, 240, 30])));
	}.play;

}
)

(
b = Buffer.alloc(s,512,1,{|z|z.sine1Msg(1.0/[1,3,5,7,9,11,13,15,17])});
c = Buffer.read(s, "/path"); // Not needed a path, but still the variable c is necessary

fork{
	s.sync;
	~sound = {arg amp = 1;
		var sig;
		var local;
		// TCHoose select randomly an output based on a (trigger, array of inputs)
		var f = [30,60,15]*Lag.kr(TChoose.kr(Impulse.kr(0.05),[0.75, 0.5, 1]), 8);

		// COsc is a chorus COsc.ar(bufnum, frequency, beat frequency, multiplier, adder)
		sig = Mix(COsc.ar(b.bufnum, f + SinOsc.ar(f*25, 0, LFTri.kr(0.01).range(0, 10)), [0.1, 0.1001, 0.2], 0.2))*0.1;


		local = sig + LocalIn.ar(2);
		15.do({
			local = AllpassN.ar(local, 0.06, Rand(0.001, 0.06), 3)
		});
		LocalOut.ar(local*0.4);

		Out.ar(0, Limiter.ar(LPF.ar(local * amp, 4000), 0.8)*EnvGen.kr(Env([0, 1, 1, 0],[9, 240, 30])));
	}.play;

}
)

(
b = Buffer.alloc(s,512,1,{|z|z.sine1Msg(1.0/[1,3,5,7,9,11,13,15,17])});
c = Buffer.read(s, "/path"); // Not needed a path, but still the variable c is necessary

fork{
	s.sync;
	~sound = {arg amp = 1;
		var sig;
		var local;
		// TCHoose select randomly an output based on a (trigger, array of inputs)
		//riducendo a 0.01 dentro impulse più piacevole
		var f = [30,60,15]*Lag.kr(TChoose.kr(Impulse.kr(0.01),[0.75, 0.5, 1]), 8);

		// COsc is a chorus COsc.ar(bufnum, frequency, beat frequency, multiplier, adder)
		//diminuendo estremo alto del range più piacevole(0,1)
		sig = Mix(COsc.ar(b.bufnum, f + SinOsc.ar(f*25, 0, LFTri.kr(0.01).range(0, 1)), [0.1, 0.1001, 0.2], 0.2))*0.1;


		// LFTri is A non-band-limited triangle oscillator with  output ranges from -1 to +1 (freq, phase, mul, add)
		//Impulse.kr(1) più piacevole
		sig = LeakDC.ar(Ringz.ar(sig, TChoose.kr(Impulse.kr(1),[88, 97, 99, 100].midicps), LFTri.kr([0.05, 0.051]).range(0.2, 0.5)));

		local = sig + LocalIn.ar(2);
		15.do({
			local = AllpassN.ar(local, 0.06, Rand(0.001, 0.06), 3)
		});
		LocalOut.ar(local*0.4);

		Out.ar(0, Limiter.ar(LPF.ar(local * amp, 4000), 0.8)*EnvGen.kr(Env([0, 1, 1, 0],[9, 240, 30])));
	}.play;

}
)

(
b = Buffer.alloc(s,512,1,{|z|z.sine1Msg(1.0/[1,3,5,7,9,11,13,15,17])});
c = Buffer.read(s, "/path"); // Not needed a path, but still the variable c is necessary

fork{
	s.sync;
	~sound = {arg amp = 1;
		var sig;
		var local;
		// TCHoose select randomly an output based on a (trigger, array of inputs)
		var f = [30,60,15]*Lag.kr(TChoose.kr(Impulse.kr(0.05),[0.75, 0.5, 1]), 8);

		// COsc is a chorus COsc.ar(bufnum, frequency, beat frequency, multiplier, adder)
		sig = Mix(COsc.ar(b.bufnum, f + SinOsc.ar(f*25, 0, LFTri.kr(0.01).range(0, 10)), [0.1, 0.1001, 0.2], 0.2))*0.1;


		// LFTri is A non-band-limited triangle oscillator with  output ranges from -1 to +1 (freq, phase, mul, add)
		sig = LeakDC.ar(Ringz.ar(sig, TChoose.kr(Impulse.kr(0.1),[88, 97, 99, 100].midicps), LFTri.kr([0.05, 0.051]).range(0.2, 0.5)));

		sig = LPF.ar(sig, 3000*LFTri.kr(0.01).range(0.1, 1));


		local = sig + LocalIn.ar(2);
		15.do({
			local = AllpassN.ar(local, 0.06, Rand(0.001, 0.06), 3)
		});
		LocalOut.ar(local*0.4);

		Out.ar(0, Limiter.ar(LPF.ar(local * amp, 4000), 0.8)*EnvGen.kr(Env([0, 1, 1, 0],[9, 240, 30])));
	}.play;

}
)


(
b = Buffer.alloc(s,512,1,{|z|z.sine1Msg(1.0/[1,3,5,7,9,11,13,15,17])});
c = Buffer.read(s, "/path"); // Not needed a path, but still the variable c is necessary

fork{
	s.sync;
	~sound = {arg amp = 1;
		var sig;
		var local;
		// TCHoose select randomly an output based on a (trigger, array of inputs)
		var f = [30,60,15]*Lag.kr(TChoose.kr(Impulse.kr(0.05),[0.75, 0.5, 1]), 8);
		// COsc is a chorus COsc.ar(bufnum, frequency, beat frequency, multiplier, adder)
		sig = Mix(COsc.ar(b.bufnum, f + SinOsc.ar(f*25, 0, LFTri.kr(0.01).range(0, 10)), [0.1, 0.1001, 0.2], 0.2))*0.1;
		sig = sig; //?
		// LeakDC is a linear filter that removes DC (in, coeff, mul, add)
		// Ringz is a 2-pole resonant filter (input, resonant frequencym, decay to -60 dB, mul, add)


		// LFTri is A non-band-limited triangle oscillator with  output ranges from -1 to +1 (freq, phase, mul, add)
		sig = LeakDC.ar(Ringz.ar(sig, TChoose.kr(Impulse.kr(0.1),[88, 97, 99, 100].midicps), LFTri.kr([0.05, 0.051]).range(0.2, 0.5)));
		// LPF is a II order LPF (input, cutoff frequency, mul, add)
		sig = LPF.ar(sig, 3000*LFTri.kr(0.01).range(0.1, 1));
		// Splay spreads an array of channels across the stereo field
		sig = sig + (Splay.ar(Array.fill(4, {PlayBuf.ar(2, c, rrand(-0.8, 0.8), loop: 2)*0.01}), 0.5));

		local = sig + LocalIn.ar(2);
		15.do({
			local = AllpassN.ar(local, 0.06, Rand(0.001, 0.06), 3)
		});
		LocalOut.ar(local*0.4);

		Out.ar(0, Limiter.ar(LPF.ar(local * amp, 4000), 0.8)*EnvGen.kr(Env([0, 1, 1, 0],[9, 240, 30])));
	}.play;

}
)


(
b = Buffer.alloc(s,512,1,{|z|z.sine1Msg(1.0/[1,3,5,7,9,11,13,15,17])});
c = Buffer.read(s, "/path"); // Not needed a path, but still the variable c is necessary

fork{
	s.sync;
	~sound = {arg amp = 1;
		var sig;
		var local;
		// TCHoose select randomly an output based on a (trigger, array of inputs)
		var f = [30,60,15]*Lag.kr(TChoose.kr(Impulse.kr(0.01),[0.75, 0.5, 1]), 8);
		// COsc is a chorus COsc.ar(bufnum, frequency, beat frequency, multiplier, adder)
		sig = Mix(COsc.ar(b.bufnum, f + SinOsc.ar(f*25, 0, LFTri.kr(0.01).range(0, 1)), [0.1, 0.1001, 0.2], 0.2))*0.1;
		sig = sig; //?
		// LeakDC is a linear filter that removes DC (in, coeff, mul, add)
		// Ringz is a 2-pole resonant filter (input, resonant frequencym, decay to -60 dB, mul, add)


		// LFTri is A non-band-limited triangle oscillator with  output ranges from -1 to +1 (freq, phase, mul, add)
		sig = LeakDC.ar(Ringz.ar(sig, TChoose.kr(Impulse.kr(1),[88, 97, 99, 100].midicps), LFTri.kr([0.05, 0.051]).range(0.2, 0.5)));
		// LPF is a II order LPF (input, cutoff frequency, mul, add)
		sig = LPF.ar(sig, 3000*LFTri.kr(0.01).range(0.1, 1));
		// Splay spreads an array of channels across the stereo field
		sig = sig + (Splay.ar(Array.fill(4, {PlayBuf.ar(2, c, rrand(-0.8, 0.8), loop: 2)*0.01}), 0.5));
		sig = CombC.ar(sig, 1.0, [0.1, 0.2], LFTri.kr(0.05).range(5, 9));

		local = sig + LocalIn.ar(2);
		15.do({
			local = AllpassN.ar(local, 0.06, Rand(0.001, 0.06), 3)
		});
		LocalOut.ar(local*0.4);

		Out.ar(0, Limiter.ar(LPF.ar(local * amp, 4000), 0.8)*EnvGen.kr(Env([0, 1, 1, 0],[9, 240, 30])));
	}.play;

}
)

(
b = Buffer.alloc(s,512,1,{|z|z.sine1Msg(1.0/[1,3,5,7,9,11,13,15,17])});
c = Buffer.read(s, "/path"); // Not needed a path, but still the variable c is necessary

fork{
	s.sync;
	~soundy = {arg amp = 1;
		var sig;
		var local;
		// TCHoose select randomly an output based on a (trigger, array of inputs)
		var f = [30,60,15]*Lag.kr(TChoose.kr(Impulse.kr(0.01),[0.75, 0.5, 1]), 8);
		// COsc is a chorus COsc.ar(bufnum, frequency, beat frequency, multiplier, adder)
		sig = Mix(COsc.ar(b.bufnum, f + SinOsc.ar(f*25, 0, LFTri.kr(0.01).range(0, 1)), [0.1, 0.1001, 0.2], 0.2))*0.1;
		sig = sig; //?
		// LeakDC is a linear filter that removes DC (in, coeff, mul, add)
		// Ringz is a 2-pole resonant filter (input, resonant frequencym, decay to -60 dB, mul, add)


		// LFTri is A non-band-limited triangle oscillator with  output ranges from -1 to +1 (freq, phase, mul, add)
		sig = LeakDC.ar(Ringz.ar(sig, TChoose.kr(Impulse.kr(1),[88, 97, 99, 100].midicps), LFTri.kr([0.05, 0.051]).range(0.2, 0.5)));
		// LPF is a II order LPF (input, cutoff frequency, mul, add)
		sig = LPF.ar(sig, 3000*LFTri.kr(0.01).range(0.1, 1));
		// Splay spreads an array of channels across the stereo field
		sig = sig + (Splay.ar(Array.fill(4, {PlayBuf.ar(2, c, rrand(-0.8, 0.8), loop: 2)*0.01}), 0.5));
		sig = CombC.ar(sig, 1.0, [0.1, 0.2], LFTri.kr(0.05).range(5, 9));

		//sig = FreeVerb.ar(sig, 0.1, 0.1, 0.5);
		sig = FreeVerb.ar(sig, 0.5, 0.1, 0.1);

		local = sig + LocalIn.ar(2);
		15.do({
			local = AllpassN.ar(local, 0.06, Rand(0.001, 0.06), 3)
		});
		LocalOut.ar(local*0.4);

		Out.ar(0, Limiter.ar(LPF.ar(local * amp, 4000), 0.8)*EnvGen.kr(Env([0, 1, 1, 0],[9, 240, 30])));
	}.play;

}
)

(
b = Buffer.alloc(s,512,1,{|z|z.sine1Msg(1.0/[1,3,5,7,9,11,13,15,17])});
c = Buffer.read(s, "/path"); // Not needed a path, but still the variable c is necessary

fork{
	s.sync;
	~soundz = {arg amp = 1;
		var sig;
		var local;
		// TCHoose select randomly an output based on a (trigger, array of inputs)
		var f = [30,60,15]*Lag.kr(TChoose.kr(Impulse.kr(0.01),[0.75, 0.5, 1]), 8);
		// COsc is a chorus COsc.ar(bufnum, frequency, beat frequency, multiplier, adder)
		sig = Mix(COsc.ar(b.bufnum, f + SinOsc.ar(f*25, 0, LFTri.kr(0.01).range(0, 1)), [0.1, 0.1001, 0.2], 0.2))*0.1;
		sig = sig; //?
		// LeakDC is a linear filter that removes DC (in, coeff, mul, add)
		// Ringz is a 2-pole resonant filter (input, resonant frequencym, decay to -60 dB, mul, add)


		// LFTri is A non-band-limited triangle oscillator with  output ranges from -1 to +1 (freq, phase, mul, add)
		sig = LeakDC.ar(Ringz.ar(sig, TChoose.kr(Impulse.kr(1),[88, 97, 99, 100].midicps), LFTri.kr([0.05, 0.051]).range(0.2, 0.5)));
		// LPF is a II order LPF (input, cutoff frequency, mul, add)
		sig = LPF.ar(sig, 3000*LFTri.kr(0.01).range(0.1, 1));
		// Splay spreads an array of channels across the stereo field
		sig = sig + (Splay.ar(Array.fill(4, {PlayBuf.ar(2, c, rrand(-0.8, 0.8), loop: 2)*0.01}), 0.5));
		sig = CombC.ar(sig, 1.0, [0.1, 0.2], LFTri.kr(0.05).range(5, 9));

		//sig = FreeVerb.ar(sig, 0.1, 0.1, 0.5);
		//sig = FreeVerb.ar(sig, 0.5, 0.1, 0.1);

		local = sig + LocalIn.ar(2);
		15.do({
			local = AllpassN.ar(local, 0.06, Rand(0.001, 0.06), 3)
		});
		LocalOut.ar(local*0.4);

		Out.ar(0, Limiter.ar(LPF.ar(local * amp, 4000), 0.8)*EnvGen.kr(Env([0, 1, 1, 0],[9, 240, 30])));
	}.play;

}
)


~sound.set(\room, 0.1);
~sound.set(\mix, 0.1);
~sound.set(\damp, 0.5);




s.quit;