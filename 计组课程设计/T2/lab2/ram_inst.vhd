ram_inst : ram PORT MAP (
		address	 => address_sig,
		clock	 => clock_sig,
		data	 => data_sig,
		wren	 => wren_sig,
		q	 => q_sig
	);
